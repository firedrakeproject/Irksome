from .constant import vecconst
from .ufl.manipulation import split_time_derivative_terms, remove_time_derivatives
from .ufl.deriv import expand_time_derivatives
from .base_time_stepper import BaseTimeStepper
from .tableaux.multistep_tableaux import MultistepTableau
from .bcs import stage2spaces4bc
from .tools import extract_timedep_arguments, replace
from ufl import lhs, Form
from ufl.constantvalue import as_ufl


class MultistepTimeStepper(BaseTimeStepper):

    """front-end class for advancing time-dependent PDE via a general multistep method

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
        F(t, u; v) == 0, where `u` is the unknown
        :class:`Function and `v` is the
        :class:TestFunction`.
    :arg method: A :class:`MultistepMethod` corresponding to the desired multistep method.
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: A :class:`Function` containing the current
            state of the problem to be solved.
    :arg bcs: An iterable of :class:`DirichletBC` containing
            the strongly-enforced boundary conditions.
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    :arg appctx: An optional :class:`dict` containing application context.
            This gets included with particular things that Irksome will
            pass into the nonlinear solver so that, say, user-defined preconditioners
            have access to it.
    :arg startup_parameters: An optional :class:`dict` used to construct a single-step TimeStepper to be used
            to find the required starting values.
    """

    def __init__(self, F, method, t, dt, u0, bcs=None, J=None, Jp=None, solver_parameters=None, bounds=None, appctx=None, nullspace=None,
                 transpose_nullspace=None, near_nullspace=None, startup_parameters=None, backend: str = "firedrake",
                 scheme_J=None, scheme_Jp=None,
                 **kwargs):

        assert isinstance(method, MultistepTableau)

        super().__init__(F, t, dt, u0,
                         bcs=bcs, appctx=appctx, nullspace=nullspace, backend=backend)
        self.num_prev_steps = len(method.b) - 1
        self.a = vecconst(method.a)
        self.b = vecconst(method.b)
        self.us = [u0.copy(deepcopy=True) for coeff in self.a[:-1]]
        self.us.append(u0)
        Fnew, bcsnew = self.get_form_and_bcs(F, t, dt, u0, self.a, self.b, bcs=bcs)
        J = self.get_bilinear_form(J, u0, method=scheme_J)
        Jp = self.get_bilinear_form(Jp, u0, method=scheme_Jp)

        self.problem = self._backend.create_variational_problem(
            Fnew, self.us[-1], J=J, Jp=Jp, bcs=bcsnew, form_compiler_parameters=kwargs.pop("form_compiler_parameters", None),
            is_linear=kwargs.pop("is_linear", False), restrict=kwargs.pop("restrict", False))

        self.solver = self._backend.create_variational_solver(
            self.problem, appctx=self.appctx, nullspace=nullspace, transpose_nullspace=transpose_nullspace,
            near_nullspace=near_nullspace, solver_parameters=solver_parameters, **kwargs)

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        self.startup_parameters = startup_parameters
        self.bounds = bounds

    # optional method to mechanically find the required starting values via a single step method
    def startup(self):
        if self.startup_parameters is None:
            return ValueError('No startup parameters provided')
        else:
            if self.num_prev_steps == 1:  # No startup required
                self.startup_t = self._backend.Constant(self.t) if isinstance(self.t, self._backend.Constant) else self.t.copy(deepcopy=True)
                return

            butcher_tableau = self.startup_parameters.get('tableau', None)
            if isinstance(butcher_tableau, MultistepTableau):
                assert butcher_tableau.num_total_steps == 2, "Cannot use a multistep method to start a multistep method"
            stepper_kwargs = self.startup_parameters.get('stepper_kwargs', {})
            num_startup_steps = self.startup_parameters.get('num_startup_steps', 1)
            assert isinstance(num_startup_steps, int) and num_startup_steps > 0

            # delayed import to avoid a circular import
            from .stepper import TimeStepper

            if isinstance(self.dt, self._backend.Constant):
                startup_dt = self._backend.Constant(self.dt / num_startup_steps)
            else:
                startup_dt = self.dt.copy(deepcopy=True)
                startup_dt.assign(startup_dt / num_startup_steps)

            self.startup_t = self._backend.Constant(self.t) if isinstance(self.t, self._backend.Constant) else self.t.copy(deepcopy=True)

            self.us[0].assign(self.u0)

            F_startup = replace(self.F, {self.t: self.startup_t})
            v, = F_startup.arguments()
            V = v.function_space()

            # grab a copy of the boundary conditions w.r.t. startup_t
            startup_bcs = []
            if self.orig_bcs is None:
                pass
            else:
                for bc in self.orig_bcs:
                    bcarg = as_ufl(bc._original_arg)
                    bcarg_startup = replace(bcarg, {self.t: self.startup_t})
                    bc_space = stage2spaces4bc(bc, V, V, 0)
                    startup_bcs.extend(bc.reconstruct(V=bc_space, g=bcarg_startup))

            self.startup_TS = TimeStepper(F_startup, butcher_tableau, self.startup_t, startup_dt, self.u0, bcs=startup_bcs, **stepper_kwargs)

            # advance the system and assign values to previous steps
            for i in range(self.num_prev_steps - 1):
                for substep in range(num_startup_steps):
                    self.startup_TS.advance()
                    self.startup_t.assign(self.startup_t + startup_dt)
                self.us[i + 1].assign(self.u0)

    def get_form_and_bcs(self, F, t, dt, u0, a, b, bcs=None):

        v, u = extract_timedep_arguments(F, u0)
        V = v.function_space()
        us = list(self.us)
        us[-1] = u0

        assert V == u0.function_space()

        split_form = split_time_derivative_terms(F, t=t, timedep_coeffs=(u,))
        F_dtless = remove_time_derivatives(split_form.time)
        F_remainder = expand_time_derivatives(split_form.remainder, t=t, timedep_coeffs=())

        # Terms with time derivatives:
        # I'm assuming we have something of the form inner(Dt(g(u0)), v)*dx.
        # Dt(g(u)) is discretised as a_s * g(u_{n+s}) + ... + a_0 * g(u_0), rather than
        # g(a_s * u_{n+s} + ... + a_0 * g(u_0)).
        Fnew = Form([])
        for (i, coeff) in enumerate(a):
            Fnew += coeff * replace(F_dtless, {u: us[i],
                                               t: t + (i - self.num_prev_steps + 1) * dt})
        # form the right hand side
        for (i, coeff) in enumerate(b):
            Fnew += dt * coeff * replace(F_remainder, {u: us[i],
                                                       t: t + (i - self.num_prev_steps + 1) * dt})
        if bcs is None:
            bcs = []
        bcsnew = []

        # grab boundary conditions at t + dt
        for bc in bcs:
            bcarg = as_ufl(bc._original_arg)
            new_bcarg = replace(bcarg, {t: t + dt})
            bc_space = stage2spaces4bc(bc, V, V, 0)
            bcsnew.extend(bc.reconstruct(V=bc_space, g=new_bcarg))

        return Fnew, bcsnew

    def get_bilinear_form(self, form, u0, method=None):
        if method is not None:
            raise NotImplementedError("Cannot change the method for the preconditioner")
        if form is None:
            return form
        _, k = extract_timedep_arguments(form, u0)
        Fbig, *_ = self.get_form_and_bcs(form, self.t, self.dt, k, self.a, self.b)
        is_bilinear = len(Fbig.arguments()) == 2
        return lhs(Fbig) if is_bilinear else self._backend.derivative(Fbig, k)

    def advance(self):
        self.solver.solve(bounds=self.bounds)

        # update previous steps
        for i in range(len(self.us) - 1):
            self.us[i].assign(self.us[i + 1])

        # update solver statistics
        self.num_steps += 1
        self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()

    def solver_stats(self):

        return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)


valid_multistep_kwargs = ("bounds", "startup_parameters")
