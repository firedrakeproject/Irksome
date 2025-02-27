# formulate RK methods to solve for stage values rather than the stage derivatives.
from functools import cached_property

import numpy as np
from FIAT import Bernstein, ufc_simplex
from firedrake import (Function, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, TestFunction, dx,
                       inner)
from firedrake.petsc import PETSc
from ufl import zero
from ufl.constantvalue import as_ufl

from .bcs import stage2spaces4bc
from .ButcherTableaux import CollocationButcherTableau
from .manipulation import extract_terms, strip_dt_form
from .tools import (AI, IA, ConstantOrZero, is_ode, replace, component_replace)
from .base_time_stepper import StageCoupledTimeStepper

vecconst = np.vectorize(ConstantOrZero)


def to_value(u0, stages, vandermonde):

    num_stages = len(stages.function_space()) // len(u0.function_space())
    Vander = vecconst(vandermonde)

    Vander_col = Vander[1:, 0]
    Vander0 = Vander[1:, 1:]

    # convert from Bernstein to Lagrange representation
    # the Bernstein coefficients are [u0; ZZ], and the Lagrange
    # are [u0; UU] since the value at the left-endpoint is unchanged.
    # Since u0 is not part of the unknown vector of stages, we disassemble
    # the Vandermonde matrix (first row is [1, 0, ...])
    v0u0 = np.reshape(np.outer(Vander_col, u0), (num_stages, *u0.ufl_shape))
    u_np = v0u0 + Vander0 @ np.reshape(stages, (num_stages, *u0.ufl_shape))
    return u_np, Vander


def getFormStage(F, butch, t, dt, u0, stages, bcs=None, splitting=None, vandermonde=None,
                 bc_constraints=None):
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
    :arg stages: a :class:`Function` referring to the stages of the time-discrete
         system at time `t`
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
    :arg bc_constraints: optionally, a dictionary mapping (some of) the boundary
         conditions in `bcs` to triples of the form (params, lower, upper) indicating
         solver parameters to use and lower and upper bounds to provide in doing
         a bounds-constrained projection of the boundary data.
         Note: if these bounds change over time, the user is responsible for maintaining
         a handle on them and updating them between time steps.
    :arg nullspace: A list of tuples of the form (index, VSB) where
         index is an index into the function space associated with `u`
         and VSB is a :class: `firedrake.VectorSpaceBasis` instance to
         be passed to a `firedrake.MixedVectorSpaceBasis` over the
         larger space associated with the Runge-Kutta method

    On output, we return a tuple consisting of several parts:

       - Fnew, the :class:`Form`
       - possibly a 4-tuple containing information needed to solve a mass matrix to update
         the solution (this is empty for RadauIIA methods for which there is a trivial
         update function.
       - UU, the :class:`firedrake.Function` holding all the stage time values.
         It lives in a :class:`firedrake.FunctionSpace` corresponding to the
         s-way tensor product of the space on which the semidiscrete
         form lives.
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the stages,
       - 'nspnew', the :class:`firedrake.MixedVectorSpaceBasis` object
         that represents the nullspace of the coupled system
    """
    v = F.arguments()[0]
    V = v.function_space()

    assert V == u0.function_space()

    num_stages = butch.num_stages

    # default to no basis transformation, identity matrix
    if vandermonde is None:
        vandermonde = np.eye(num_stages + 1)

    # s-way product space for the stage variables
    Vbig = stages.function_space()
    VV = TestFunction(Vbig)

    # set up the pieces we need to work with to do our substitutions
    v_np = np.reshape(VV, (num_stages, *u0.ufl_shape))
    u_np, Vander = to_value(u0, stages, vandermonde)

    Vander_inv = vecconst(np.linalg.inv(vandermonde))

    C = vecconst(butch.c)
    A = vecconst(butch.A)

    split_form = extract_terms(F)

    Fnew = zero()

    # first, process terms with a time derivative.  I'm
    # assuming we have something of the form inner(Dt(g(u0)), v)*dx
    # For each stage i, this gets replaced with
    # inner((g(stages[i]) - g(u0))/dt, v)*dx
    # but we have to carefully handle the cases where g indexes into
    # pieces of u
    dtless = strip_dt_form(split_form.time)

    if splitting is None or splitting == AI:
        # time derivative part
        for i in range(num_stages):
            repl = {t: t+C[i]*dt,
                    u0: u_np[i] - u0,
                    v: v_np[i]}

            Fnew += component_replace(dtless, repl)

        # Now for the non-time derivative parts
        for i in range(num_stages):
            # replace test function
            repl = {v: v_np[i]}
            Ftmp = component_replace(split_form.remainder, repl)

            # replace the solution with stage values
            for j in range(num_stages):
                repl = {t: t + C[j] * dt,
                        u0: u_np[j]}

                # and sum the contribution
                Fnew += A[i, j] * dt * component_replace(Ftmp, repl)

    elif splitting == IA:
        Ainv = vecconst(np.linalg.inv(butch.A))

        # time derivative part gets inverse of Butcher matrix.
        for i in range(num_stages):
            repl = {v: v_np[i]}
            Ftmp = component_replace(dtless, repl)

            for j in range(num_stages):
                repl = {t: t + C[j] * dt,
                        u0: u_np[j] - u0}

                Fnew += Ainv[i, j] * component_replace(Ftmp, repl)
        # rest of the operator: just diagonal!
        for i in range(num_stages):
            repl = {t: t+C[i]*dt,
                    u0: u_np[i],
                    v: v_np[i]}

            Fnew += dt * replace(split_form.remainder, repl)
    else:
        raise NotImplementedError("Can't handle that splitting type")

    if bcs is None:
        bcs = []
    if bc_constraints is None:
        bc_constraints = {}
    bcsnew = []

    # For each BC, we need a new BC for each stage
    # so we need to figure out how the function is indexed (mixed + vec)
    # and then set it to have the value of the original argument at
    # time t+C[i]*dt.

    for bc in bcs:
        bcarg = as_ufl(bc._original_arg)

        if bc in bc_constraints:
            bcparams, bclower, bcupper = bc_constraints[bc]
            gcur = replace(bcarg, {t: t+C[i] * dt})
            gcur = gcur - Vander[1+i, 0] * bcarg
        else:
            gdats_cur = np.zeros((num_stages,), dtype="O")
            for i in range(num_stages):
                Vbigi = stage2spaces4bc(bc, V, Vbig, i)
                gcur = replace(bcarg, {t: t+C[i]*dt})
                gcur = gcur - Vander[1+i, 0] * bcarg
                gdats_cur[i] = gcur

            zdats_cur = Vander_inv[1:, 1:] @ gdats_cur

            bcnew_cur = []
            for i in range(num_stages):
                Vbigi = stage2spaces4bc(bc, V, Vbig, i)
                bcnew_cur.append(bc.reconstruct(V=Vbigi, g=zdats_cur[i]))

            bcsnew.extend(bcnew_cur)
    return Fnew, bcsnew


class StageValueTimeStepper(StageCoupledTimeStepper):
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None, update_solver_parameters=None,
                 bc_constraints=None,
                 splitting=AI, basis_type=None,
                 nullspace=None, appctx=None):

        # we can only do DAE-type problems correctly if one assumes a stiffly-accurate method.
        assert is_ode(F, u0) or butcher_tableau.is_stiffly_accurate

        self.butcher_tableau = butcher_tableau
        self.bc_constraints = bc_constraints

        degree = butcher_tableau.num_stages

        if basis_type is None:
            vandermonde = np.eye(degree+1)
        elif basis_type == "Bernstein":
            assert isinstance(butcher_tableau, CollocationButcherTableau), "Need collocation for Bernstein conversion"
            bern = Bernstein(ufc_simplex(1), degree)
            cc = np.reshape(np.append(0, butcher_tableau.c), (-1, 1))
            vandermonde = bern.tabulate(0, np.reshape(cc, (-1, 1)))[(0, )].T
        else:
            raise ValueError("Unknown or unimplemented basis transformation type")
        self.vandermonde = vandermonde

        super().__init__(F, t, dt, u0, butcher_tableau.num_stages, bcs=bcs,
                         solver_parameters=solver_parameters,
                         appctx=appctx, nullspace=nullspace,
                         splitting=splitting, butcher_tableau=butcher_tableau)

        self.num_fields = len(u0.function_space())

        if (not butcher_tableau.is_stiffly_accurate) and (basis_type != "Bernstein"):
            unew, Fupdate, update_bcs = self.update_stuff
            self.update_problem = NonlinearVariationalProblem(
                Fupdate, unew, update_bcs)

            self.update_solver = NonlinearVariationalSolver(
                self.update_problem,
                solver_parameters=update_solver_parameters)
            self._update = self._update_general
        else:
            self._update = self._update_stiff_acc

        # stash these for later in case we do bounds constraints
        self.stage_lower_bound = Function(self.stages.function_space())
        self.stage_upper_bound = Function(self.stages.function_space())

    def _update_stiff_acc(self):
        u0 = self.u0
        u0bits = u0.subfunctions
        UUs = self.stages.subfunctions

        for i, u0bit in enumerate(u0bits):
            u0bit.assign(UUs[self.num_fields*(self.num_stages-1)+i])

    @cached_property
    def update_stuff(self):
        # only form update stuff if we need it
        # which means neither stiffly accurate nor Vandermonde
        unew = Function(self.V)
        v, = self.F.arguments()
        Fupdate = inner(unew - self.u0, v) * dx

        C = vecconst(self.butcher_tableau.c)
        B = vecconst(self.butcher_tableau.b)
        t = self.t
        dt = self.dt
        u0 = self.u0
        split_form = extract_terms(self.F)
        u_np, _ = to_value(self.u0, self.stages, self.vandermonde)

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
        return unew, Fupdate, update_bcs

    def _update_general(self):
        unew, Fupdate, update_bcs = self.update_stuff
        self.update_solver.solve()
        unewbits = unew.subfunctions
        for u0bit, unewbit in zip(self.u0.subfunctions, unewbits):
            u0bit.assign(unewbit)

    def advance(self, bounds=None):
        if bounds is None:
            stage_bounds = None
        else:
            bounds_type, lower, upper = bounds
            slb = self.stage_lower_bound
            sub = self.stage_upper_bound
            if bounds_type == "stage":
                if lower is None:
                    slb.assign(PETSc.NINFINITY)
                else:
                    for i in range(self.num_stages):
                        for j, lower_bit in enumerate(lower.subfunctions):
                            slb.subfunctions[i*self.num_fields+j].assign(lower_bit)
                if upper is None:
                    sub.assign(PETSc.INFINITY)
                else:
                    for i in range(self.num_stages):
                        for j, upper_bit in enumerate(upper.subfunctions):
                            sub.subfunctions[i*self.num_fields+j].assign(upper_bit)
            elif bounds_type == "last_stage":
                if lower is None:
                    slb.assign(PETSc.NINFINITY)
                else:
                    for i in range(self.num_stages-1):
                        for j in range(self.num_fields):
                            slb.subfunctions[i*self.num_fields+j].assign(PETSc.NINFINITY)
                    for j, lower_bit in enumerate(lower.subfunctions):
                        slb.subfunctions[-(self.num_fields-j)].assign(lower_bit)
                if upper is None:
                    sub.assign(PETSc.INFINITY)
                else:
                    for i in range(self.num_stages-1):
                        for j in range(self.num_fields):
                            sub.subfunctions[i*self.num_fields+j].assign(PETSc.INFINITY)
                    for j, upper_bit in enumerate(upper.subfunctions):
                        sub.subfunctions[-(self.num_fields-j)].assign(upper_bit)
            else:
                raise ValueError("Unknown bounds type")

            stage_bounds = (slb, sub)

        self.solver.solve(bounds=stage_bounds)

        self.num_steps += 1
        self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()

        self._update()

    def get_form_and_bcs(self, stages, butcher_tableau=None):
        if butcher_tableau is None:
            butcher_tableau = self.butcher_tableau
        return getFormStage(self.F, butcher_tableau,
                            self.t, self.dt, self.u0,
                            stages, bcs=self.orig_bcs,
                            splitting=self.splitting,
                            vandermonde=self.vandermonde,
                            bc_constraints=self.bc_constraints)
