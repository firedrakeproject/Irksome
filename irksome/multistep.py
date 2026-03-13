from .tools import replace, vecconst
from .manipulation import extract_terms
from .deriv import expand_time_derivatives, TimeDerivative
from .base_time_stepper import BaseTimeStepper
from .multistep_tableaux import MultistepTableau
from ufl.constantvalue import as_ufl
from firedrake import NonlinearVariationalProblem, NonlinearVariationalSolver, derivative


class MultistepTimeStepper(BaseTimeStepper):

    """front-end class for advancing time-dependent PDE via a general multistep method

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
        F(t, u; v) == 0, where `u` is the unknown
        :class:`firedrake.Function and `v` is the
        :class:firedrake.TestFunction`.
    :arg method: A :class:`MultistepMethod` corresponding to the desired multistep method.
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing
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

    def __init__(self, F, method, t, dt, u0, bcs=None, Fp=None, solver_parameters=None, bounds=None, appctx=None, nullspace=None,
                 transpose_nullspace=None, near_nullspace=None, startup_parameters=None, **kwargs):

        assert isinstance(method, MultistepTableau)

        super().__init__(F, t, dt, u0,
                         bcs=bcs, appctx=appctx, nullspace=nullspace)

        self.num_prev_steps = len(method.b) - 1
        self.a = vecconst(method.a)
        self.b = vecconst(method.b)
        self.us = [u0.copy(deepcopy=True) for coeff in self.a[:-1]]
        self.us.append(u0)
        Fnew, bcsnew = self.get_form_and_bcs(F, t, dt, u0, self.a, self.b, bcs=bcs)

        if Fp is not None:
            Fpnew, _ = self.get_form_and_bcs(Fp, t, dt, u0, self.a, self.b, bcs=bcs)
            Jp = derivative(Fpnew, self.us[-1])
        else:
            Jp = None

        self.problem = NonlinearVariationalProblem(Fnew, self.us[-1], J=Jp, bcs=bcsnew, form_compiler_parameters=kwargs.pop("form_compiler_parameters", None),
                                                   is_linear=kwargs.pop("is_linear", False),
                                                   restrict=kwargs.pop("restrict", False))

        self.solver = NonlinearVariationalSolver(self.problem, appctx=self.appctx,
                                                 nullspace=nullspace,
                                                 transpose_nullspace=transpose_nullspace,
                                                 near_nullspace=near_nullspace,
                                                 solver_parameters=solver_parameters,
                                                 **kwargs
                                                 )

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        self.F = F
        self.t = t
        self.dt = dt
        self.u0 = u0
        self.startup_parameters = startup_parameters
        self.bcs = bcs
        self.bounds = bounds

    # optional method to mechanically find the required starting values via a single step method
    # WARNING: This overwrites the value of t.
    def startup(self):

        if self.startup_parameters is None:
            return ValueError('No startup parameters provided')
        else:
            if self.num_prev_steps == 1:  # No startup required
                return

            butcher_tableau = self.startup_parameters.get('tableau', None)
            if isinstance(butcher_tableau, MultistepTableau):
                assert butcher_tableau.num_total_steps == 2, "Cannot use a multistep method to start a multistep method"
            stepper_kwargs = self.startup_parameters.get('stepper_kwargs', {})
            startup_dt_div = self.startup_parameters.get('dt_div', 1)
            assert isinstance(startup_dt_div, int) and startup_dt_div > 0

            # delayed import to avoid a circular import
            from .stepper import TimeStepper

            self.us[0].assign(self.u0)
            self.TS = TimeStepper(self.F, butcher_tableau, self.t, self.dt, self.u0, bcs=self.bcs, **stepper_kwargs)

            # modify the timestep
            self.dt.assign(self.dt / startup_dt_div)

            # advance the system and assign values to previous steps
            for i in range(self.num_prev_steps - 1):
                for substep in range(startup_dt_div):
                    self.TS.advance()
                    self.t.assign(self.t + self.dt)
                self.us[i + 1].assign(self.u0)
            self.dt.assign(self.dt * startup_dt_div)

    def get_form_and_bcs(self, F, t, dt, u0, a, b, bcs=None):

        F = expand_time_derivatives(F, t=t, timedep_coeffs=(u0, ))
        v, = F.arguments()
        V = v.function_space()

        assert V == u0.function_space()

        split_form = extract_terms(F)
        F_dt = split_form.time
        F_remainder = split_form.remainder

        # replace the time derivative with a linear combination of the previous steps
        temp_form = 0.0
        for (i, coeff) in enumerate(a):
            temp_form += coeff * self.us[i]

        dtu = TimeDerivative(u0)
        Fnew = replace(F_dt, {dtu: temp_form})

        # form the right hand side
        for (i, coeff) in enumerate(b):
            Fnew += dt * coeff * replace(F_remainder, {u0: self.us[i],
                                                       t: t + (i - self.num_prev_steps + 1) * dt})
        if bcs is None:
            bcs = []
        bcsnew = []

        # grab boundary conditions at t + dt
        for bc in bcs:
            g0 = as_ufl(bc._original_arg)
            g0new = replace(g0, {t: t + dt})
            bcsnew.append(bc.reconstruct(V=V, g=g0new))

        return Fnew, bcsnew

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


valid_multistep_kwargs = ("Fp", "bounds", "startup_parameters")
