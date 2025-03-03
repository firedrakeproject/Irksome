import numpy
from firedrake import Function, TestFunction
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake import assemble, dx, inner, norm

from ufl import diff, zero
from ufl.algorithms import expand_derivatives
from ufl.constantvalue import as_ufl
from .tools import component_replace, replace, AI, vecconst
from .deriv import TimeDerivative  # , apply_time_derivatives
from .bcs import EmbeddedBCData, BCStageData, bc2space
from .manipulation import extract_terms
from .base_time_stepper import StageCoupledTimeStepper


def getForm(F, butch, t, dt, u0, stages, bcs=None, bc_type=None, splitting=AI):
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
    :arg splitting: a callable that maps the (floating point) Butcher matrix
         a to a pair of matrices `A1, A2` such that `butch.A = A1 A2`.  This is used
         to vary between the classical RK formulation and Butcher's reformulation
         that leads to a denser mass matrix with block-diagonal stiffness.
         Some choices of function will assume that `butch.A` is invertible.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg stages: a :class:`Function` representing the stages to be solved for.
    :arg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
         containing (possibly time-dependent) boundary conditions imposed
         on the system.
    :arg bc_type: How to manipulate the strongly-enforced boundary
         conditions to derive the stage boundary conditions.  Should
         be a string, either "DAE", which implements BCs as
         constraints in the style of a differential-algebraic
         equation, or "ODE", which takes the time derivative of the
         boundary data and evaluates this for the stage values

    On output, we return a tuple consisting of four parts:

       - Fnew, the :class:`Form`
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the stages,
    """
    if bc_type is None:
        bc_type = "DAE"
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
    w_np = numpy.reshape(stages, (num_stages, *u0.ufl_shape))
    A1w = A1 @ w_np
    A2invw = A2inv @ w_np

    dtu = TimeDerivative(u0)
    Fnew = zero()
    for i in range(num_stages):
        repl = {t: t + c[i] * dt,
                v: v_np[i],
                u0: u0 + A1w[i] * dt,
                dtu: A2invw[i]}
        Fnew += component_replace(F, repl)

    if bcs is None:
        bcs = []
    if bc_type == "ODE":
        assert splitting == AI, "ODE-type BC aren't implemented for this splitting strategy"

        def bc2gcur(bc, i):
            gorig = as_ufl(bc._original_arg)
            gfoo = expand_derivatives(diff(gorig, t))
            return replace(gfoo, {t: t + c[i] * dt})

    elif bc_type == "DAE":
        try:
            bA1inv = numpy.linalg.inv(bA1)
            A1inv = vecconst(bA1inv)
        except numpy.linalg.LinAlgError:
            raise NotImplementedError("Cannot have DAE BCs for this Butcher Tableau/splitting")

        def bc2gcur(bc, i):
            gorig = as_ufl(bc._original_arg)
            ucur = bc2space(bc, u0)
            gcur = (1/dt) * sum((replace(gorig, {t: t + c[j]*dt}) - ucur) * A1inv[i, j]
                                for j in range(num_stages))
            return gcur
    else:
        raise ValueError("Unrecognised bc_type: %s", bc_type)

    # This logic uses information set up in the previous section to
    # set up the new BCs for either method
    bcnew = []
    for bc in bcs:
        for i in range(num_stages):
            gcur = bc2gcur(bc, i)
            bcnew.append(BCStageData(bc, gcur, u0, stages, i))

    return Fnew, bcnew


class StageDerivativeTimeStepper(StageCoupledTimeStepper):
    """Front-end class for advancing a time-dependent PDE via a Runge-Kutta
    method formulated in terms of stage derivatives.

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
            the Runge-Kutta method to be used for time marching.
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing
            the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the RK method.
    :arg bc_type: How to manipulate the strongly-enforced boundary
            conditions to derive the stage boundary conditions.
            Should be a string, either "DAE", which implements BCs as
            constraints in the style of a differential-algebraic
            equation, or "ODE", which takes the time derivative of the
            boundary data and evaluates this for the stage values
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    :arg splitting: An callable used to factor the Butcher matrix
    :arg appctx: An optional :class:`dict` containing application context.
            This gets included with particular things that Irksome will
            pass into the nonlinear solver so that, say, user-defined preconditioners
            have access to it.
    :arg nullspace: A list of tuples of the form (index, VSB) where
            index is an index into the function space associated with
            `u` and VSB is a :class: `firedrake.VectorSpaceBasis`
            instance to be passed to a
            `firedrake.MixedVectorSpaceBasis` over the larger space
            associated with the Runge-Kutta method
    """
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None, splitting=AI,
                 appctx=None, nullspace=None, bc_type="DAE"):

        self.num_fields = len(u0.function_space())
        self.butcher_tableau = butcher_tableau
        A1, A2 = splitting(butcher_tableau.A)
        try:
            self.updateb = vecconst(numpy.linalg.solve(A2.T, butcher_tableau.b))
        except numpy.linalg.LinAlgError:
            raise NotImplementedError("A=A1 A2 splitting needs A2 invertible")

        super().__init__(F, t, dt, u0,
                         butcher_tableau.num_stages, bcs=bcs,
                         solver_parameters=solver_parameters,
                         appctx=appctx, nullspace=nullspace,
                         splitting=splitting, bc_type=bc_type,
                         butcher_tableau=butcher_tableau)

    def _update(self):
        """Assuming the algebraic problem for the RK stages has been
        solved, updates the solution.  This will not typically be
        called by an end user."""
        b = self.updateb
        dt = self.dt
        ns = self.num_stages
        nf = self.num_fields

        # Note: this now cates the optimized/stiffly accurate case as b[s] == Zero() will get dropped
        for i, u0bit in enumerate(self.u0.subfunctions):
            u0bit += sum(self.stages.subfunctions[nf * s + i] * (b[s] * dt) for s in range(ns))

    def get_form_and_bcs(self, stages, butcher_tableau=None):
        if butcher_tableau is None:
            butcher_tableau = self.butcher_tableau
        return getForm(self.F, butcher_tableau, self.t, self.dt,
                       self.u0, stages, self.orig_bcs, self.bc_type,
                       self.splitting)


class AdaptiveTimeStepper(StageDerivativeTimeStepper):
    """Front-end class for advancing a time-dependent PDE via an adaptive
    Runge-Kutta method.

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
            the Runge-Kutta method to be used for time marching.
    :arg t: A :class:`firedrake.Constant` instance that always
            contains the time value at the beginning of a time step
    :arg dt: A :class:`firedrake.Constant` containing the size of the
            current time step.  The user may adjust this value between
            time steps; however, note that the adaptive time step
            controls may adjust this before the step is taken.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg tol: The temporal truncation error tolerance
    :arg dtmin: Minimal acceptable time step.  An exception is raised
            if the step size drops below this threshhold.
    :arg dtmax: Maximal acceptable time step, imposed as a hard cap;
            this can be adjusted externally once the time-stepper is
            instantiated, by modifying `stepper.dt_max`
    :arg KI: Integration gain for step-size controller.  Should be less
            than 1/p, where p is the expected order of the scheme.  Larger
            values lead to faster (attempted) increases in time-step size
            when steps are accepted.  See Gustafsson, Lundh, and Soderlind,
            BIT 1988.
    :arg KP: Proportional gain for step-size controller. Controls dependence
            on ratio of (error estimate)/(step size) in determining new
            time-step size when steps are accepted.  See Gustafsson, Lundh,
            and Soderlind, BIT 1988.
    :arg max_reject: Maximum number of rejected timesteps in a row that
            does not lead to a failure
    :arg onscale_factor: Allowable tolerance in determining initial
            timestep to be "on scale"
    :arg safety_factor: Safety factor used when shrinking timestep if
            a proposed step is rejected
    :arg gamma0_params: Solver parameters for mass matrix solve when using
            an embedded scheme with explicit first stage
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing
            the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the RK method.
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    :arg nullspace: A list of tuples of the form (index, VSB) where
            index is an index into the function space associated with
            `u` and VSB is a :class: `firedrake.VectorSpaceBasis`
            instance to be passed to a
            `firedrake.MixedVectorSpaceBasis` over the larger space
            associated with the Runge-Kutta method
    """
    def __init__(self, F, butcher_tableau, t, dt, u0,
                 bcs=None, appctx=None, solver_parameters=None,
                 bc_type="DAE", splitting=AI, nullspace=None,
                 tol=1.e-3, dtmin=1.e-15, dtmax=1.0, KI=1/15, KP=0.13,
                 max_reject=10, onscale_factor=1.2, safety_factor=0.9,
                 gamma0_params=None):
        assert butcher_tableau.btilde is not None
        super(AdaptiveTimeStepper, self).__init__(F, butcher_tableau,
                                                  t, dt, u0, bcs=bcs, appctx=appctx, solver_parameters=solver_parameters,
                                                  bc_type=bc_type, splitting=splitting, nullspace=nullspace)

        from firedrake.petsc import PETSc
        self.print = PETSc.Sys.Print

        self.dt_min = dtmin
        self.dt_max = dtmax
        self.dt_old = 0.0

        self.delb = butcher_tableau.btilde - butcher_tableau.b
        self.gamma0 = butcher_tableau.gamma0
        self.gamma0_params = gamma0_params
        self.KI = KI
        self.KP = KP
        self.max_reject = max_reject
        self.onscale_factor = onscale_factor
        self.safety_factor = safety_factor

        self.error_func = Function(u0.function_space())
        self.tol = tol
        self.err_old = 0.0
        self.contreject = 0

        split_form = extract_terms(F)
        self.dtless_form = -split_form.remainder

        # Set up and cache boundary conditions for error estimate
        embbc = []
        if self.gamma0 != 0:
            # Grab spaces for BCs
            embbc = [EmbeddedBCData(bc, butcher_tableau, self.t, self.dt, self.u0, self.stages)
                     for bc in bcs]
        self.embbc = embbc

    def _estimate_error(self):
        """Assuming that the RK stages have been evaluated, estimates
        the temporal truncation error by taking the norm of the
        difference between the new solutions computed by the two
        methods.  Typically will not be called by the end user."""
        dtc = float(self.dt)
        delb = self.delb
        ws = self.stages.subfunctions
        nf = self.num_fields
        ns = self.num_stages
        u0 = self.u0

        # Initialize e to be gamma*h*f(old value of u)
        error_func = Function(u0.function_space())
        # Only do the hard stuff if gamma0 is not zero
        if self.gamma0 != 0.0:
            error_test = TestFunction(u0.function_space())
            f_form = inner(error_func, error_test)*dx-self.gamma0*dtc*self.dtless_form
            f_problem = NLVP(f_form, error_func, bcs=self.embbc)
            f_solver = NLVS(f_problem, solver_parameters=self.gamma0_params)
            f_solver.solve()

        # Accumulate delta-b terms over stages
        error_func_bits = error_func.subfunctions
        for s in range(ns):
            for i, e in enumerate(error_func_bits):
                e += dtc*float(delb[s])*ws[nf*s+i]
        return norm(assemble(error_func))

    def advance(self):
        """Attempts to advances the system from time `t` to time `t +
        dt`.  If the error threshhold is exceeded, will adaptively
        decrease the time step until the step is accepted.  Also
        predicts new time step once the step is accepted.
        Note: overwrites the value `u0`."""
        if float(self.dt) > self.dt_max:
            self.dt.assign(self.dt_max)
        self.print("\tTrying dt = %e" % (float(self.dt)))
        while 1:
            self.solver.solve()
            self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
            self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()
            err_current = float(self._estimate_error())
            err_old = float(self.err_old)
            dt_old = float(self.dt_old)
            dt_current = float(self.dt)
            tol = float(self.tol)
            dt_pred = dt_current*((dt_current*tol)/err_current)**(1/self.butcher_tableau.embedded_order)
            self.print("\tTruncation error is %e" % (err_current))

            # Rejected step shrinks the time-step
            if err_current >= dt_current*tol:
                dtnew = dt_current*(self.safety_factor*dt_current*tol/err_current)**(1./self.butcher_tableau.embedded_order)
                self.print("\tShrinking time-step to %e" % (dtnew))
                self.dt.assign(dtnew)
                self.contreject += 1
                if dtnew <= self.dt_min or numpy.isfinite(dtnew) is False:
                    raise RuntimeError("The time-step became an invalid number.")
                if self.contreject >= self.max_reject:
                    raise RuntimeError(f"The time-step was rejected {self.max_reject} times in a row. Please increase the tolerance or decrease the starting time-step.")

            # Initial time-step selector
            elif self.num_steps == 0 and dt_current < self.dt_max and dt_pred > self.onscale_factor*dt_current and self.contreject <= self.max_reject:

                # Increase the initial time-step
                dtnew = min(dt_pred, self.dt_max)
                self.print("\tIncreasing time-step to %e" % (dtnew))
                self.dt.assign(dtnew)
                self.contreject += 1

            # Accepted step increases the time-step
            else:
                if dt_old != 0.0 and err_old != 0.0 and dt_current < self.dt_max:
                    dtnew = min(dt_current*((dt_current*tol)/err_current)**self.KI*(err_old/err_current)**self.KP*(dt_current/dt_old)**self.KP, self.dt_max)
                    self.print("\tThe step was accepted and the new time-step is %e" % (dtnew))
                else:
                    dtnew = min(dt_current, self.dt_max)
                    self.print("\tThe step was accepted and the time-step remains at %e " % (dtnew))
                self._update()
                self.contreject = 0
                self.num_steps += 1
                self.dt_old = self.dt
                self.dt.assign(dtnew)
                self.err_old = err_current
                return (err_current, dt_current)
