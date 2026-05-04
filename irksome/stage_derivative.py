import numpy
from firedrake import Function, TestFunction
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake import assemble, dx, inner, norm

from .constant import vecconst
from .tools import AI
from .ufl.deriv import expand_time_derivatives
from .bcs import EmbeddedBCData
from .base_time_stepper import StageCoupledTimeStepper
from .form_manipulation import getForm
from .butcher_tableaux import CollocationButcherTableau
from FIAT import ufc_simplex
from FIAT.barycentric_interpolation import LagrangePolynomialSet
from .ufl.manipulation import split_time_derivative_terms



class StageDerivativeTimeStepper(StageCoupledTimeStepper):
    """Front-end class for advancing a time-dependent PDE via a Runge-Kutta
    method formulated in terms of stage derivatives.

    :arg F: a :class:`ufl.Form` instance describing the semi-discrete problem.
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
        the Runge-Kutta method to be used for time marching.
    :arg t: a :class:`firedrake.Constant` or :class:`firedrake.Function`
        on the Real space over the same mesh as ``u0``.  This serves as
        a variable referring to the current time.
    :arg dt: a :class:`firedrake.Constant` or :class:`firedrake.Function`
        on the Real space over the same mesh as ``u0``.  This serves as
        a variable referring to the current time step size.
    :arg u0: A :class:`firedrake.Function` containing the current
        state of the problem to be solved.
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` or :class:`firedrake.EquationBC`
        containing the strongly-enforced boundary conditions.  Irksome will
        manipulate these to obtain boundary conditions for each
        stage of the RK method.
    :arg bc_type: How to manipulate the strongly-enforced boundary
        conditions to derive the stage boundary conditions.
        Should be a string, either "DAE", which implements BCs as
        constraints in the style of a differential-algebraic
        equation, or "ODE", which takes the time derivative of the
        boundary data and evaluates this for the stage values.
        Support for :class:`firedrake.EquationBC` in `bcs` is limited
        to DAE style BCs.
    :arg solver_parameters: A :class:`dict` of solver parameters that
        will be used in solving the algebraic problem associated
        with each time step.
    :arg splitting: An callable used to factor the Butcher matrix
    :arg appctx: An optional :class:`dict` containing application context.
        This gets included with particular things that Irksome will
        pass into the nonlinear solver so that, say, user-defined preconditioners
        have access to it.
    :arg nullspace: An optional nullspace object.
    :kwarg sample_points: An optional kwarg used to evaluate collocation methods
        at additional points in time.
    """
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None, splitting=AI,
                 appctx=None, bc_type="DAE", aux_indices=None, sample_points=None, **kwargs):

        self.num_fields = len(u0.function_space())
        self.butcher_tableau = butcher_tableau
        A1, A2 = splitting(butcher_tableau.A)
        try:
            self.updateb = vecconst(numpy.linalg.solve(A2.T, butcher_tableau.b))
        except numpy.linalg.LinAlgError:
            raise NotImplementedError("A=A1 A2 splitting needs A2 invertible")

        self.aux_indices = aux_indices
        super().__init__(F, t, dt, u0,
                         butcher_tableau.num_stages, bcs=bcs,
                         solver_parameters=solver_parameters,
                         appctx=appctx,
                         splitting=splitting, bc_type=bc_type,
                         butcher_tableau=butcher_tableau,
                         sample_points=sample_points, **kwargs)

    def _update(self):
        """Assuming the algebraic problem for the RK stages has been
        solved, updates the solution.  This will not typically be
        called by an end user."""
        b = self.updateb
        dt = self.dt
        ns = self.num_stages
        nf = self.num_fields

        # Note: this now catches the optimized/stiffly accurate case as b[s] == Zero() will get dropped

        for i, u0bit in enumerate(self.u0.subfunctions):
            u0bit += sum(self.stages.subfunctions[nf * s + i] * (b[s] * dt) for s in range(ns))

    def get_form_and_bcs(self, stages, F=None, bcs=None, tableau=None):
        if bcs is None:
            bcs = self.orig_bcs
        return getForm(F or self.F,
                       tableau or self.butcher_tableau,
                       self.t, self.dt,
                       self.u0, stages, bcs, self.bc_type,
                       splitting=self.splitting,
                       aux_indices=self.aux_indices)

    def tabulate_poly(self, sample_points):
        if not isinstance(self.butcher_tableau, CollocationButcherTableau):
            raise ValueError("Need a collocation method to evaluate the collocation polynomial")
        nodes = numpy.insert(self.butcher_tableau.c, 0, 0.0)
        if len(set(nodes)) != len(nodes):
            raise ValueError("Need non-confluent collocation method for polynomial evaluation")

        ref_el = ufc_simplex(1)
        lag_basis = LagrangePolynomialSet(ref_el, nodes)
        evaluation_vander = vecconst(lag_basis.tabulate(sample_points, 0)[(0,)])

        butcher_A = vecconst(self.butcher_tableau.A)
        num_terms = self.num_stages + 1
        coeffs = vecconst(numpy.zeros((num_terms, num_terms)))
        coeffs[0, :] = 1
        coeffs[1:, 1:] = self.dt * butcher_A.T
        vander = coeffs @ evaluation_vander
        return vander


class AdaptiveTimeStepper(StageDerivativeTimeStepper):
    """Front-end class for advancing a time-dependent PDE via an adaptive
    Runge-Kutta method.

    :arg F: a :class:`ufl.Form` instance describing the semi-discrete problem.
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
        the Runge-Kutta method to be used for time marching.
    :arg t: a :class:`firedrake.Constant` or :class:`firedrake.Function`
        on the Real space over the same mesh as `u0`.  This serves as
        a variable referring to the current time.
    :arg dt: a :class:`firedrake.Constant` or :class:`firedrake.Function`
        on the Real space over the same mesh as `u0`.  This serves as
        a variable referring to the current time step size.
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
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` or :class:`EquationBC`
        containing the strongly-enforced boundary conditions.  Irksome will
        manipulate these to obtain boundary conditions for each
        stage of the RK method.
    :arg solver_parameters: A :class:`dict` of solver parameters that
        will be used in solving the algebraic problem associated
        with each time step.
    :arg nullspace: An optional nullspace object.
    """
    def __init__(self, F, butcher_tableau, t, dt, u0,
                 bcs=None, appctx=None, solver_parameters=None,
                 bc_type="DAE", splitting=AI, nullspace=None,
                 tol=1.e-3, dtmin=1.e-15, dtmax=1.0, KI=1/15, KP=0.13,
                 max_reject=10, onscale_factor=1.2, safety_factor=0.9,
                 gamma0_params=None, **kwargs):
        assert butcher_tableau.btilde is not None
        super(AdaptiveTimeStepper, self).__init__(F, butcher_tableau,
                                                  t, dt, u0, bcs=bcs, appctx=appctx, solver_parameters=solver_parameters,
                                                  bc_type=bc_type, splitting=splitting, nullspace=nullspace, **kwargs)

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

        split_form = split_time_derivative_terms(F, t=t, timedep_coeffs=(u0,))
        F_remainder = expand_time_derivatives(split_form.remainder, t=t, timedep_coeffs=())
        self.dtless_form = -F_remainder

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
