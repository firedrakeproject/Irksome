# formulate RK methods to solve for stage values rather than the stage derivatives.
import numpy
from FIAT import Bernstein, ufc_simplex
from FIAT.barycentric_interpolation import LagrangePolynomialSet
from firedrake import (Function, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, TestFunction, dx,
                       inner)
from ufl import as_tensor, Form
from ufl.constantvalue import as_ufl

from .bcs import stage2spaces4bc
from .tableaux.ButcherTableaux import CollocationButcherTableau
from .ufl.deriv import expand_time_derivatives
from .ufl.manipulation import (has_composite_time_derivative,
                               split_time_derivative_terms,
                               remove_time_derivatives)
from .tools import AI, dot, reshape, replace
from .constant import vecconst
from .base_time_stepper import StageCoupledTimeStepper


def to_value(u0, stages, vandermonde):
    """convert from Bernstein to Lagrange representation

    the Bernstein coefficients are [u0; ZZ], and the Lagrange
    are [u0; UU] since the value at the left-endpoint is unchanged.
    Since u0 is not part of the unknown vector of stages, we disassemble
    the Vandermonde matrix (first row is [1, 0, ...]).
    """
    ZZ_np = reshape(stages, (-1, *u0.ufl_shape))
    if vandermonde is None:
        return ZZ_np
    u0_np = reshape(u0, (-1, *u0.ufl_shape))
    u_np = numpy.concatenate((u0_np, ZZ_np))
    return dot(vandermonde[1:], u_np)


def getFormStage(F, butch, t, dt, u0, stages, bcs=None, splitting=AI, vandermonde=None, aux_indices=None):
    """Given a time-dependent variational form and a
    :class:`ButcherTableau`, produce UFL for the s-stage RK method.

    :arg F: a :class:`ufl.Form` instance describing the semi-discrete problem.
    :arg butch: the :class:`ButcherTableau` for the RK method being used to
        advance in time.
    :arg t: a :class:`firedrake.Constant` or :class:`firedrake.Function`
        on the Real space over the same mesh as `u0`.  This serves as
        a variable referring to the current time.
    :arg dt: a :class:`firedrake.Constant` or :class:`firedrake.Function`
        on the Real space over the same mesh as `u0`.  This serves as
        a variable referring to the current time step size.
        The user may adjust this value between time steps.
    :arg u0: a :class:`Function` referring to the state of
        the PDE system at time `t`
    :arg stages: a :class:`Function` representing the stages to be solved for.
        It lives in a :class:`firedrake.FunctionSpace` corresponding to the
        s-way tensor product of the space on which the semidiscrete
        form lives.
    :kwarg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
        containing (possibly time-dependent) boundary conditions imposed
        on the system.
    :kwarg splitting: a callable that maps the (floating point) Butcher matrix
        a to a pair of matrices `A1, A2` such that `butch.A = A1 A2`.  This is used
        to vary between the classical RK formulation and Butcher's reformulation
        that leads to a denser mass matrix with block-diagonal stiffness.
        Only `AI` and `IA` are currently supported.
    :kwarg vandermonde: a numpy array encoding a change of basis to the Lagrange
        polynomials associated with the collocation nodes from some other
        (e.g. Bernstein or Chebyshev) basis.  This allows us to solve for the
        coefficients in some basis rather than the values at particular stages,
        which can be useful for satisfying bounds constraints.
        If none is provided, we assume it is the identity, working in the
        Lagrange basis.
    :kwarg aux_indices: a list of field indices, currently ignored.

    :returns: a 2-tuple of
       - `Fnew`, the :class:`Form`
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the stages
    """
    v, = F.arguments()
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
    v_np = reshape(test, (num_stages, *v.ufl_shape))
    w_np = to_value(u0, stages, vandermonde)
    A1Tv = dot(A1.T, v_np)
    A2invTv = dot(A2inv.T, v_np)

    # first, process terms with a time derivative.  I'm
    # assuming we have something of the form inner(Dt(g(u0)), v)*dx
    # For each stage i, this gets replaced with
    # inner((g(stages[i]) - g(u0))/dt, v)*dx
    split_form = split_time_derivative_terms(F, t=t, timedep_coeffs=(u0,))
    F_dtless = remove_time_derivatives(split_form.time)
    F_remainder = expand_time_derivatives(split_form.remainder, t=t, timedep_coeffs=())

    Fnew = Form([])
    # Terms with time derivatives: use two evaluations so that
    # Dt(g(u)) is discretised as g(U_i) - g(u0), not g(U_i - u0).
    # These are identical for linear g but differ for nonlinear g,
    # and the two-evaluation form is what gives mass conservation.
    for i in range(num_stages):
        repl_new = {t: t + c[i] * dt,
                    v: A2invTv[i],
                    u0: w_np[i]}
        # Evaluate g at the old solution u0 (not substituted) and
        # old time t (not substituted).
        repl_old = {v: A2invTv[i]}
        Fnew += replace(F_dtless, repl_new) - replace(F_dtless, repl_old)

    # Handle the rest of the terms
    for i in range(num_stages):
        # replace the solution with stage values
        repl = {t: t + c[i] * dt,
                v: A1Tv[i] * dt,
                u0: w_np[i]}
        Fnew += replace(F_remainder, repl)

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
            bcsnew.extend(bc.reconstruct(V=Vbigi, g=as_tensor(g_np[i])))
    return Fnew, bcsnew


class StageValueTimeStepper(StageCoupledTimeStepper):
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None,
                 update_solver_parameters=None,
                 splitting=AI, basis_type=None,
                 appctx=None, bounds=None,
                 use_collocation_update=False,
                 **kwargs):

        self.num_fields = len(u0.function_space())
        self.butcher_tableau = butcher_tableau
        self.basis_type = basis_type

        degree = butcher_tableau.num_stages

        if basis_type is None or basis_type == 'Lagrange':
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
                         appctx=appctx,
                         splitting=splitting, butcher_tableau=butcher_tableau, bounds=bounds,
                         **kwargs)

        # Conservative variational update inherits the stage solve's
        # parameters by default: same form structure, same Jacobian
        # sparsity, so the same preconditioner is the right starting
        # point.  Callers who want a different setup can pass
        # update_solver_parameters explicitly.
        if update_solver_parameters is None:
            update_solver_parameters = solver_parameters

        self.set_initial_guess()

        if use_collocation_update:
            # Use the terminal value of the collocation polynomial to update the solution. Note: collocation update is only implemented for constant-in-time boundary conditions.
            # TODO: create an assertion to check for constant-in-time boundary conditions.
            nodes = numpy.insert(self.butcher_tableau.c, 0, 0.0)

            assert isinstance(self.butcher_tableau, CollocationButcherTableau), "Need a collocation method for collocation update"
            assert (self.basis_type is None or self.basis_type == "Lagrange"), "Collocation update requires the Lagrange form of the collocation polynomial"
            assert (len(set(nodes)) == self.butcher_tableau.num_stages + 1), "Need a non-confluent collocation method to use collocation update"

            # The collocation update evaluates the collocation polynomial
            # at t=1 as a linear combination of u0 and the stages.  That
            # combination is not mass-conservative when Dt(g(u)) has a
            # nonlinear g, because g(linear combo) != linear combo of
            # g(stage_i).  Refuse rather than silently returning a
            # non-conservative answer; users can disable use_collocation_update
            # to fall back to the conservative variational update path.
            if has_composite_time_derivative(F, u0):
                raise NotImplementedError(
                    "use_collocation_update=True is incompatible with "
                    "Dt(g(u)) for a nonlinear g of the prognostic variable: "
                    "the collocation polynomial's terminal value is a "
                    "linear combination of stages and is not "
                    "mass-conservative.  Disable use_collocation_update "
                    "to use the conservative variational update."
                )

            lag_basis = LagrangePolynomialSet(ufc_simplex(1), nodes)
            collocation_vander = vecconst(lag_basis.tabulate((1.0,))[(0,)])

            self.collocation_vander = collocation_vander
            self._update = self._update_collocation

        elif (not butcher_tableau.is_stiffly_accurate) and (basis_type != "Bernstein"):
            # For composite Dt(g(u)) we need the conservative variational
            # update; the bAinv linear combination of stages is correct
            # for g = identity but breaks for nonlinear g.  For the
            # linear case we keep the bAinv shortcut: it is conservative
            # by construction (g = id makes both formulations agree)
            # AND it handles DAEs correctly, which the conservative
            # variational update does not (it leaves the algebraic
            # components of u_new unconstrained, producing a singular
            # update Jacobian).
            if has_composite_time_derivative(F, u0):
                # Note: composite Dt(g(u)) on a DAE form (where some
                # component of u0 has no time evolution) is not
                # supported by this path -- the conservative variational
                # update leaves the algebraic block unconstrained, and
                # SNES will fail with a singular linear solve.  We do
                # not detect this up-front because the available
                # syntactic checks (is_ode etc.) are confused by the
                # chain-rule promotion of Dt(u[i]) -> Dt(u)[i].  Stiffly
                # accurate methods (RadauIIA, BackwardEuler) avoid this
                # since u_new = U_s and no update solve is needed.
                self.unew, self.update_solver = self.get_update_solver(update_solver_parameters)
                self._update = self._update_general
            else:
                try:
                    A = butcher_tableau.A
                    b = butcher_tableau.b
                    self.bAinv = vecconst(numpy.linalg.solve(A.T, b))
                    self.update_scale = 1-numpy.sum(self.bAinv)
                    self._update = self._update_Ainv
                except numpy.linalg.LinAlgError:
                    self.unew, self.update_solver = self.get_update_solver(update_solver_parameters)
                    self._update = self._update_general
        else:
            self._update = self._update_stiff_acc

    def _update_Ainv(self):
        nf = self.num_fields
        ns = self.num_stages
        scale = self.update_scale
        bAinv = self.bAinv
        for i, u0bit in enumerate(self.u0.subfunctions):
            u0bit *= scale
            u0bit += sum(self.stages.subfunctions[nf * s + i] * bAinv[s] for s in range(ns))

    def _update_stiff_acc(self):
        for i, u0bit in enumerate(self.u0.subfunctions):
            u0bit.assign(self.stages.subfunctions[self.num_fields*(self.num_stages-1)+i])

    def get_update_solver(self, update_solver_parameters):
        # Build a conservative variational update for u_new from the
        # stage values.  The head is the same two-evaluation form used
        # in the stage equations, evaluated at unew and u0:
        #
        #     replace(F_dtless, {u0: unew}) - replace(F_dtless, {u0: u0})
        #
        # which expands to inner(g(unew) - g(u0), v)*dx for the typical
        # mass term inner(Dt(g(u)), v)*dx.  For g = identity this is
        # exactly inner(unew - u0, v)*dx -- the previous head -- so the
        # discrete update equation is unchanged in the linear case.
        F = self.F
        t = self.t
        dt = self.dt
        u0 = self.u0
        unew = Function(u0.function_space())

        split_form = split_time_derivative_terms(F, t=t, timedep_coeffs=(u0,))
        F_dtless = remove_time_derivatives(split_form.time)
        F_remainder = expand_time_derivatives(split_form.remainder, t=t, timedep_coeffs=())

        # Two-evaluation conservative head.  Subtracting the two
        # replaced forms keeps the test function v unchanged in both,
        # so the result is a single Form valid for assembly.
        Fupdate = replace(F_dtless, {u0: unew}) - replace(F_dtless, {u0: u0})

        C = vecconst(self.butcher_tableau.c)
        B = vecconst(self.butcher_tableau.b)
        u_np = to_value(self.u0, self.stages, self.vandermonde)

        for i in range(self.num_stages):
            repl = {t: t + C[i] * dt,
                    u0: u_np[i]}
            Fupdate += dt * B[i] * replace(F_remainder, repl)

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
        # Seed unew with u0 before solving.  The conservative head's
        # Jacobian is the moisture-capacity-weighted mass matrix
        # C(unew)*v*phi*dx, and for soils like Haverkamp the capacity
        # vanishes at h = 0 -- so a fresh Function (zero-initialised)
        # gives a singular initial Jacobian and SNES fails before
        # taking any Newton step.  u0 is the natural warm start anyway.
        self.unew.assign(self.u0)
        self.update_solver.solve()
        self.u0.assign(self.unew)

    def _update_collocation(self):
        stage_vals = numpy.array(self.u0.subfunctions + self.stages.subfunctions, dtype=object)
        for i, u0bit in enumerate(self.u0.subfunctions):
            u0bit.assign(stage_vals[i::self.num_fields] @ self.collocation_vander)

    def get_form_and_bcs(self, stages, F=None, bcs=None, tableau=None):
        if bcs is None:
            bcs = self.orig_bcs
        return getFormStage(F or self.F,
                            tableau or self.butcher_tableau,
                            self.t, self.dt, self.u0,
                            stages, bcs=bcs,
                            splitting=self.splitting,
                            vandermonde=self.vandermonde)

    def set_initial_guess(self):
        """Set a constant-in-time initial guess"""
        for k in range(self.num_stages):
            for i, u0bit in enumerate(self.u0.subfunctions):
                sbit = self.stages.subfunctions[self.num_fields * k + i]
                sbit.assign(u0bit)
