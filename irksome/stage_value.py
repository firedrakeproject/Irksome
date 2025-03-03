# formulate RK methods to solve for stage values rather than the stage derivatives.
import numpy
from FIAT import Bernstein, ufc_simplex
from firedrake import (Function, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, TestFunction, dx,
                       inner)
from ufl import zero
from ufl.constantvalue import as_ufl

from .bcs import stage2spaces4bc
from .ButcherTableaux import CollocationButcherTableau
from .manipulation import extract_terms, strip_dt_form
from .tools import AI, is_ode, replace, component_replace, vecconst
from .base_time_stepper import StageCoupledTimeStepper


def to_value(u0, stages, vandermonde):
    """convert from Bernstein to Lagrange representation

    the Bernstein coefficients are [u0; ZZ], and the Lagrange
    are [u0; UU] since the value at the left-endpoint is unchanged.
    Since u0 is not part of the unknown vector of stages, we disassemble
    the Vandermonde matrix (first row is [1, 0, ...]).
    """
    ZZ_np = numpy.reshape(stages, (-1, *u0.ufl_shape))
    if vandermonde is None:
        return ZZ_np
    u0_np = numpy.reshape(u0, (-1, *u0.ufl_shape))
    u_np = numpy.concatenate((u0_np, ZZ_np))
    return vandermonde[1:] @ u_np


def getFormStage(F, butch, t, dt, u0, stages, bcs=None, splitting=None, vandermonde=None):
    """Given a time-dependent variational form and a
    :class:`ButcherTableau`, produce UFL for the s-stage RK method.

    :arg F: UFL form for the semidiscrete ODE/DAE
    :arg butch: the :class:`ButcherTableau` for the RK method being used to
         advance in time.
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg stages: a :class:`Function` representing the stages to be solved for.
         It lives in a :class:`firedrake.FunctionSpace` corresponding to the
         s-way tensor product of the space on which the semidiscrete
         form lives.
    :arg splitting: a callable that maps the (floating point) Butcher matrix
         a to a pair of matrices `A1, A2` such that `butch.A = A1 A2`.  This is used
         to vary between the classical RK formulation and Butcher's reformulation
         that leads to a denser mass matrix with block-diagonal stiffness.
         Only `AI` and `IA` are currently supported.
    :arg vandermonde: a numpy array encoding a change of basis to the Lagrange
         polynomials associated with the collocation nodes from some other
         (e.g. Bernstein or Chebyshev) basis.  This allows us to solve for the
         coefficients in some basis rather than the values at particular stages,
         which can be useful for satisfying bounds constraints.
         If none is provided, we assume it is the identity, working in the
         Lagrange basis.
    :arg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
         containing (possibly time-dependent) boundary conditions imposed
         on the system.
    :arg nullspace: A list of tuples of the form (index, VSB) where
         index is an index into the function space associated with `u`
         and VSB is a :class: `firedrake.VectorSpaceBasis` instance to
         be passed to a `firedrake.MixedVectorSpaceBasis` over the
         larger space associated with the Runge-Kutta method

    On output, we return a tuple consisting of several parts:

       - `Fnew`, the :class:`Form`
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the stages,
    """
    v = F.arguments()[0]
    V = v.function_space()
    assert V == u0.function_space()

    c = vecconst(butch.c)
    bA1, bA2 = splitting(butch.A)
    try:
        bA2inv = numpy.linalg.inv(bA2)
    except numpy.linalg.LinAlgError:
        raise NotImplementedError("We require A = A1 A2 with A2 invertible")
    A1 = vecconst(bA1)
    A2inv = vecconst(bA2inv)

    # s-way product space for the stage variables
    num_stages = butch.num_stages
    Vbig = stages.function_space()
    test = TestFunction(Vbig)

    # set up the pieces we need to work with to do our substitutions
    v_np = numpy.reshape(test, (num_stages, *u0.ufl_shape))
    w_np = to_value(u0, stages, vandermonde)
    A1Tv = A1.T @ v_np
    A2invTv = A2inv.T @ v_np

    # first, process terms with a time derivative.  I'm
    # assuming we have something of the form inner(Dt(g(u0)), v)*dx
    # For each stage i, this gets replaced with
    # inner((g(stages[i]) - g(u0))/dt, v)*dx
    split_form = extract_terms(F)
    F_dtless = strip_dt_form(split_form.time)
    F_remainder = split_form.remainder

    Fnew = zero()
    # Terms with time derivatives
    for i in range(num_stages):
        repl = {t: t + c[i] * dt,
                v: A2invTv[i],
                u0: w_np[i] - u0}
        Fnew += component_replace(F_dtless, repl)

    # Handle the rest of the terms
    for i in range(num_stages):
        # replace the solution with stage values
        repl = {t: t + c[i] * dt,
                v: A1Tv[i] * dt,
                u0: w_np[i]}
        Fnew += component_replace(F_remainder, repl)

    if bcs is None:
        bcs = []
    bcsnew = []

    if vandermonde is not None:
        Vander_inv = vecconst(numpy.linalg.inv(vandermonde.astype(float)))

    # For each BC, we need a new BC for each stage
    # so we need to figure out how the function is indexed (mixed + vec)
    # and then set it to have the value of the original argument at
    # time t+C[i]*dt.
    for bc in bcs:
        bcarg = as_ufl(bc._original_arg)
        g_np = numpy.array([replace(bcarg, {t: t + ci * dt}) for ci in c])
        if vandermonde is not None:
            g_np -= vandermonde[1:, 0] * bcarg
            g_np = Vander_inv[1:, 1:] @ g_np

        for i in range(num_stages):
            Vbigi = stage2spaces4bc(bc, V, Vbig, i)
            bcsnew.extend(bc.reconstruct(V=Vbigi, g=g_np[i]))
    return Fnew, bcsnew


class StageValueTimeStepper(StageCoupledTimeStepper):
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None, update_solver_parameters=None,
                 splitting=AI, basis_type=None,
                 nullspace=None, appctx=None, bounds=None):

        # we can only do DAE-type problems correctly if one assumes a stiffly-accurate method.
        assert is_ode(F, u0) or butcher_tableau.is_stiffly_accurate

        self.num_fields = len(u0.function_space())
        self.butcher_tableau = butcher_tableau

        degree = butcher_tableau.num_stages

        if basis_type is None:
            vandermonde = None
        elif basis_type == "Bernstein":
            assert isinstance(butcher_tableau, CollocationButcherTableau), "Need collocation for Bernstein conversion"
            bern = Bernstein(ufc_simplex(1), degree)
            pts = numpy.reshape(numpy.append(0, butcher_tableau.c), (-1, 1))
            vandermonde = bern.tabulate(0, pts)[(0, )].T
        else:
            raise ValueError("Unknown or unimplemented basis transformation type")

        if vandermonde is not None:
            vandermonde = vecconst(vandermonde)
        self.vandermonde = vandermonde

        super().__init__(F, t, dt, u0, butcher_tableau.num_stages, bcs=bcs,
                         solver_parameters=solver_parameters,
                         appctx=appctx, nullspace=nullspace,
                         splitting=splitting, butcher_tableau=butcher_tableau, bounds=bounds)

        if (not butcher_tableau.is_stiffly_accurate) and (basis_type != "Bernstein"):
            self.unew, self.update_solver = self.get_update_solver(update_solver_parameters)
            self._update = self._update_general
        else:
            self._update = self._update_stiff_acc

    def _update_stiff_acc(self):
        for i, u0bit in enumerate(self.u0.subfunctions):
            u0bit.assign(self.stages.subfunctions[self.num_fields*(self.num_stages-1)+i])

    def get_update_solver(self, update_solver_parameters):
        # only form update stuff if we need it
        # which means neither stiffly accurate nor Vandermonde
        unew = Function(self.u0.function_space())
        v, = self.F.arguments()
        Fupdate = inner(unew - self.u0, v) * dx

        C = vecconst(self.butcher_tableau.c)
        B = vecconst(self.butcher_tableau.b)
        t = self.t
        dt = self.dt
        u0 = self.u0
        split_form = extract_terms(self.F)
        u_np = to_value(self.u0, self.stages, self.vandermonde)

        for i in range(self.num_stages):
            repl = {t: t + C[i] * dt,
                    u0: u_np[i]}
            Fupdate += dt * B[i] * component_replace(split_form.remainder, repl)

        # And the BC's for the update -- just the original BC at t+dt
        update_bcs = []
        for bc in self.orig_bcs:
            bcarg = as_ufl(bc._original_arg)
            gcur = replace(bcarg, {t: t + dt})
            update_bcs.append(bc.reconstruct(g=gcur))

        update_problem = NonlinearVariationalProblem(
            Fupdate, unew, update_bcs)

        update_solver = NonlinearVariationalSolver(
            update_problem,
            solver_parameters=update_solver_parameters)

        return unew, update_solver

    def _update_general(self):
        self.update_solver.solve()
        self.u0.assign(self.unew)

    def get_form_and_bcs(self, stages, butcher_tableau=None):
        if butcher_tableau is None:
            butcher_tableau = self.butcher_tableau
        return getFormStage(self.F, butcher_tableau,
                            self.t, self.dt, self.u0,
                            stages, bcs=self.orig_bcs,
                            splitting=self.splitting,
                            vandermonde=self.vandermonde)
