import numpy
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake import (DirichletBC, Function, TestFunction,
                       inner, dx, norm, assemble, project, replace)
from firedrake.__future__ import interpolate
from firedrake.dmhooks import pop_parent, push_parent
from ufl.constantvalue import as_ufl
from .dirk_stepper import DIRKTimeStepper
from .explicit_stepper import ExplicitTimeStepper
from .getForm import AI, getForm
from .stage import StageValueTimeStepper
from .imex import RadauIIAIMEXMethod
from .manipulation import extract_terms


def TimeStepper(F, butcher_tableau, t, dt, u0, **kwargs):
    """Helper function to dispatch between various back-end classes
       for doing time stepping.  Returns an instance of the
       appropriate class.

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` iss the
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
    :arg nullspace: A list of tuples of the form (index, VSB) where
            index is an index into the function space associated with
            `u` and VSB is a :class: `firedrake.VectorSpaceBasis`
            instance to be passed to a
            `firedrake.MixedVectorSpaceBasis` over the larger space
            associated with the Runge-Kutta method
    :arg stage_type: Whether to formulate in terms of a stage
            derivatives or stage values.
    :arg splitting: An callable used to factor the Butcher matrix
    :arg bc_type: For stage derivative formulation, how to manipulate
            the strongly-enforced boundary conditions.
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    :arg update_solver_parameters: A :class:`dict` of parameters for
            inverting the mass matrix at each step (only used if
            stage_type is "value")
    :arg adaptive_parameters: A :class:`dict` of parameters for use with
            adaptive time stepping (only used if stage_type is "deriv")
    """

    valid_kwargs_per_stage_type = {
        "deriv": ["stage_type", "bcs", "nullspace", "solver_parameters", "appctx",
                  "bc_type", "splitting", "adaptive_parameters"],
        "value": ["stage_type", "bcs", "nullspace", "solver_parameters",
                  "update_solver_parameters", "appctx", "splitting"],
        "dirk": ["stage_type", "bcs", "nullspace", "solver_parameters", "appctx"],
        "explicit": ["stage_type", "bcs", "solver_parameters", "appctx"],
        "imex": ["Fexp", "stage_type", "bcs", "nullspace",
                 "it_solver_parameters", "prop_solver_parameters",
                 "splitting", "appctx",
                 "num_its_initial", "num_its_per_step"]}

    valid_adapt_parameters = ["tol", "dtmin", "dtmax", "KI", "KP",
                              "max_reject", "onscale_factor",
                              "safety_factor", "gamma0_params"]

    stage_type = kwargs.get("stage_type", "deriv")
    adapt_params = kwargs.get("adaptive_parameters")
    if adapt_params is not None:
        assert stage_type == "deriv", "Adaptive time stepping is only implemented for derivative stage type"
    for cur_kwarg in kwargs.keys():
        if cur_kwarg not in valid_kwargs_per_stage_type:
            assert cur_kwarg in valid_kwargs_per_stage_type[stage_type]

    if stage_type == "deriv":
        bcs = kwargs.get("bcs")
        bc_type = kwargs.get("bc_type", "DAE")
        splitting = kwargs.get("splitting", AI)
        appctx = kwargs.get("appctx")
        solver_parameters = kwargs.get("solver_parameters")
        nullspace = kwargs.get("nullspace")
        if adapt_params is None:
            return StageDerivativeTimeStepper(
                F, butcher_tableau, t, dt, u0, bcs, appctx=appctx,
                solver_parameters=solver_parameters, nullspace=nullspace,
                bc_type=bc_type, splitting=splitting)
        else:
            for param in adapt_params:
                assert param in valid_adapt_parameters
            tol = adapt_params.get("tol", 1e-3)
            dtmin = adapt_params.get("dtmin", 1.e-15)
            dtmax = adapt_params.get("dtmax", 1.0)
            KI = adapt_params.get("KI", 1/15)
            KP = adapt_params.get("KP", 0.13)
            max_reject = adapt_params.get("max_reject", 10)
            onscale_factor = adapt_params.get("onscale_factor", 1.2)
            safety_factor = adapt_params.get("safety_factor", 0.9)
            gamma0_params = adapt_params.get("gamma0_params")
        return AdaptiveTimeStepper(
            F, butcher_tableau, t, dt, u0, bcs, appctx=appctx,
            solver_parameters=solver_parameters, nullspace=nullspace,
            bc_type=bc_type, splitting=splitting,
            tol=tol, dtmin=dtmin, dtmax=dtmax, KI=KI, KP=KP,
            max_reject=max_reject, onscale_factor=onscale_factor,
            safety_factor=safety_factor, gamma0_params=gamma0_params)
    elif stage_type == "value":
        bcs = kwargs.get("bcs")
        splitting = kwargs.get("splitting", AI)
        appctx = kwargs.get("appctx")
        solver_parameters = kwargs.get("solver_parameters")
        update_solver_parameters = kwargs.get("update_solver_parameters")
        nullspace = kwargs.get("nullspace")
        return StageValueTimeStepper(
            F, butcher_tableau, t, dt, u0, bcs=bcs, appctx=appctx,
            solver_parameters=solver_parameters,
            splitting=splitting,
            update_solver_parameters=update_solver_parameters,
            nullspace=nullspace)
    elif stage_type == "dirk":
        bcs = kwargs.get("bcs")
        appctx = kwargs.get("appctx")
        solver_parameters = kwargs.get("solver_parameters")
        nullspace = kwargs.get("nullspace")
        return DIRKTimeStepper(
            F, butcher_tableau, t, dt, u0, bcs,
            solver_parameters, appctx, nullspace)
    elif stage_type == "explicit":
        bcs = kwargs.get("bcs")
        appctx = kwargs.get("appctx")
        solver_parameters = kwargs.get("solver_parameters")
        return ExplicitTimeStepper(
            F, butcher_tableau, t, dt, u0, bcs,
            solver_parameters, appctx)
    elif stage_type == "imex":
        Fexp = kwargs.get("Fexp")
        assert Fexp is not None
        bcs = kwargs.get("bcs")
        appctx = kwargs.get("appctx")
        splitting = kwargs.get("splitting", AI)
        it_solver_parameters = kwargs.get("it_solver_parameters")
        prop_solver_parameters = kwargs.get("prop_solver_parameters")
        nullspace = kwargs.get("nullspace")
        num_its_initial = kwargs.get("num_its_initial", 0)
        num_its_per_step = kwargs.get("num_its_per_step", 0)

        return RadauIIAIMEXMethod(
            F, Fexp, butcher_tableau, t, dt, u0, bcs,
            it_solver_parameters, prop_solver_parameters,
            splitting, appctx, nullspace,
            num_its_initial, num_its_per_step)


class StageDerivativeTimeStepper:
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
        self.u0 = u0
        self.F = F
        self.orig_bcs = bcs
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = len(butcher_tableau.b)
        self.butcher_tableau = butcher_tableau
        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        bigF, stages, bigBCs, bigNSP, bigBCdata = \
            getForm(F, butcher_tableau, t, dt, u0, bcs, bc_type, splitting, nullspace)

        self.stages = stages
        self.bigBCs = bigBCs
        self.bigBCdata = bigBCdata
        problem = NLVP(bigF, stages, bigBCs)
        appctx_irksome = {"F": F,
                          "butcher_tableau": butcher_tableau,
                          "t": t,
                          "dt": dt,
                          "u0": u0,
                          "bcs": bcs,
                          "bc_type": bc_type,
                          "splitting": splitting,
                          "nullspace": nullspace}
        if appctx is None:
            appctx = appctx_irksome
        else:
            appctx = {**appctx, **appctx_irksome}

        push_parent(u0.function_space().dm, stages.function_space().dm)
        self.solver = NLVS(problem,
                           appctx=appctx,
                           solver_parameters=solver_parameters,
                           nullspace=bigNSP)
        pop_parent(u0.function_space().dm, stages.function_space().dm)

        if self.num_stages == 1 and self.num_fields == 1:
            self.ws = (stages,)
        else:
            self.ws = stages.subfunctions

        A1, A2 = splitting(butcher_tableau.A)
        try:
            self.updateb = numpy.linalg.solve(A2.T, butcher_tableau.b)
        except numpy.linalg.LinAlgError:
            raise NotImplementedError("A=A1 A2 splitting needs A2 invertible")
        boo = numpy.zeros(self.updateb.shape, dtype=self.updateb.dtype)
        boo[-1] = 1
        if numpy.allclose(self.updateb, boo):
            self._update = self._update_A2Tmb
        else:
            self._update = self._update_general

    def _update_general(self):
        """Assuming the algebraic problem for the RK stages has been
        solved, updates the solution.  This will not typically be
        called by an end user."""
        b = self.updateb
        dtc = float(self.dt)
        u0 = self.u0
        ns = self.num_stages
        nf = self.num_fields

        ws = self.ws
        u0bits = u0.subfunctions
        for s in range(ns):
            for i, u0bit in enumerate(u0bits):
                u0bit += dtc * float(b[s]) * ws[nf*s+i]

    def _update_A2Tmb(self):
        """Assuming the algebraic problem for the RK stages has been
        solved, updates the solution.  This will not typically be
        called by an end user.  This handles the common but highly
        specialized case of `w = Ak` or `A = I A` splitting where
        A2^{-T} b = e_{num_stages}"""
        dtc = float(self.dt)
        u0 = self.u0
        ns = self.num_stages
        nf = self.num_fields

        ws = self.ws
        u0bits = u0.subfunctions
        for i, u0bit in enumerate(u0bits):
            u0bit += dtc * ws[nf*(ns-1)+i]

    def advance(self):
        """Advances the system from time `t` to time `t + dt`.
        Note: overwrites the value `u0`."""
        for gdat, gcur, gmethod in self.bigBCdata:
            gmethod(gcur, self.u0)

        push_parent(self.u0.function_space().dm, self.stages.function_space().dm)
        self.solver.solve()
        pop_parent(self.u0.function_space().dm, self.stages.function_space().dm)

        self.num_steps += 1
        self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()
        self._update()

    def solver_stats(self):
        return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)


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
        self.print = lambda x: PETSc.Sys.Print(x)

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
        if self.gamma0 != 0:
            # Grab spaces for BCs
            v = F.arguments()[0]
            V = v.function_space()
            num_fields = len(V)
            num_stages = butcher_tableau.num_stages
            btilde = butcher_tableau.btilde
            ws = self.ws

            class EmbeddedBCData(object):
                def __init__(self, bc, t, dt, num_fields, num_stages, btilde, V, ws, u0):
                    gorig = as_ufl(bc._original_arg)
                    gcur = replace(gorig, {t: t+dt})
                    if num_fields == 1:  # not mixed space
                        comp = bc.function_space().component
                        if comp is not None:  # check for sub-piece of vector-valued
                            Vsp = V.sub(comp)
                            # for j in range(num_stages):
                            #     gcur -= dt*btilde[j]*ws[j].sub(comp)
                            try:
                                gdat = assemble(interpolate(gcur-u0.sub(comp), Vsp))
                                def gmethod(g, u):
                                    gdat.interpolate(g-u.sub(comp))
                                    for j in range(num_stages):
                                        gdat -= dt * btilde[j] * ws[j].sub(comp)
                                    return gdat
                                # gmethod = lambda g, u: gdat.interpolate(g-u.sub(comp))
                            except:  # noqa: E722
                                gdat = project(gcur-u0.sub(comp), Vsp)
                                def gmethod(g, u):
                                    gdat.project(g-u.sub(comp))
                                    for j in range(num_stages):
                                        gdat -= dt * btilde[j] * ws[j].sub(comp)
                                    return gdat
                                # gmethod = lambda g, u: gdat.project(g-u.sub(comp))
                        else:
                            Vsp = V
                            # for j in range(num_stages):
                            #     gcur -= dt*btilde[j]*ws[j]
                            try:
                                gdat = assemble(interpolate(gcur-u0, Vsp))
                                def gmethod(g, y):
                                    gdat.interpolate(g-u)
                                    for j in range(num_steps):
                                        gdat -= dt * btilde[j] * ws[j]
                                    return gdat
                                # gmethod = lambda g, u: gdat.interpolate(g-u)
                            except:  # noqa: E722
                                gdat = project(gcur-u0, Vsp)
                                def gmethod(g, y):
                                    gdat.project(g-u)
                                    for j in range(num_steps):
                                        gdat -= dt * btilde[j] * ws[j]
                                    return gdat                               
                                gmethod = lambda g, u: gdat.project(g-u)
                    else:  # mixed space
                        sub = bc.function_space_index()
                        comp = bc.function_space().component
                        if comp is not None:  # check for sub-piece of vector-valued
                            Vsp = V.sub(sub).sub(comp)
                            # for j in range(num_stages):
                            #     gcur -= dt*btilde[j]*ws[num_fields*j+sub].sub(comp)
                            try:
                                gdat = assemble(interpolate(gcur-u0.sub(sub).sub(comp), Vsp))
                                def gmethod(g, u):
                                    gdat.interpolate(g-u.sub(sub).sub(comp))
                                    for j in range(num_stages):
                                        gdat -= dt * btilde[j] * ws[num_fields*j+sub].sub(comp)
                                    return gdat
                            except:  # noqa: E722
                                gdat = project(gcur-u0.sub(sub).sub(comp), Vsp)
                                def gmethod(g, u):
                                    gdat.project(g-u.sub(sub).sub(comp))
                                    for j in range(num_stages):
                                        gdat -= dt * btilde[j] * ws[num_fields*j+sub].sub(comp)
                                    return gdat
                        else:
                            Vsp = V.sub(sub)
                            # for j in range(num_stages):
                            #     gcur -= dt*btilde[j]*ws[num_fields*j+sub]
                            try:
                                gdat = assemble(interpolate(gcur-u0.sub(sub), Vsp))
                                def gmethod(g, u):
                                    gdat.interpolate(g-u.sub(sub))
                                    for j in range(num_stages):
                                        gdat -= dt * btilde[j] * ws[num_fields*j+sub]
                                    return gdat
                                
                            except:  # noqa: E722
                                gdat = project(gcur-u0.sub(sub), Vsp)
                                def gmethod(g, u):
                                    gdat.project(g-u.sub(sub))
                                    for j in range(num_stages):
                                        gdat -= dt * btilde[j] * ws[num_fields*j+sub]
                                    return gdat                                

                    self.gstuff = (gdat, gcur, gmethod, Vsp)

            embbc = []
            gblah = []
            for bc in bcs:
                blah = EmbeddedBCData(bc, self.t, self.dt, num_fields, num_stages, btilde, V, ws, self.u0)
                gdat, gcur, gmethod, gVsp = blah.gstuff
                gblah.append((gdat, gcur, gmethod))
                embbc.append(DirichletBC(gVsp, gdat, bc.sub_domain))
            self.embbc = embbc
            self.gblah = gblah

    def _estimate_error(self):
        """Assuming that the RK stages have been evaluated, estimates
        the temporal truncation error by taking the norm of the
        difference between the new solutions computed by the two
        methods.  Typically will not be called by the end user."""
        dtc = float(self.dt)
        delb = self.delb
        ws = self.ws
        nf = self.num_fields
        ns = self.num_stages
        u0 = self.u0

        # Initialize e to be gamma*h*f(old value of u)
        error_func = Function(u0.function_space())
        # Only do the hard stuff if gamma0 is not zero
        if self.gamma0 != 0.0:
            error_test = TestFunction(u0.function_space())
            f_form = inner(error_func, error_test)*dx-self.gamma0*dtc*self.dtless_form
            for gdat, gcur, gmethod in self.gblah:
                gmethod(gcur, self.u0)
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
            for gdat, gcur, gmethod in self.bigBCdata:
                gmethod(gcur, self.u0)

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
