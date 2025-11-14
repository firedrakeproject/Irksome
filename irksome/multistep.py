from .tools import replace, vecconst, getNullspace
from .manipulation import extract_terms, strip_dt_form
from .deriv import expand_time_derivatives, TimeDerivative
from .stepper import TimeStepper, valid_base_kwargs
from .base_time_stepper import BaseTimeStepper
from ufl.constantvalue import as_ufl
from ufl import zero
from firedrake import NonlinearVariationalProblem, NonlinearVariationalSolver
from .MultistepMethods import multistep_dict

class MultistepStepper(BaseTimeStepper):
    
    """front-end class for advancing time-dependent PDE via the BDF2 Method
    
    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
        F(t, u; v) == 0, where `u` is the unknown
        :class:`firedrake.Function and `v` is the
        :class:firedrake.TestFunction`.
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg method: A string corresponding to a standard method OR a tuple of arrays, (a, b), containing the coefficients defining the method.
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
    :arg startup_params: An optional :class:`dict` used to construct a single-step TimeStepper to be used 
            to find the required starting values.
    """

    def __init__(self, F, t, dt, u0, method, bcs=None, solver_parameters=None, bounds=None, appctx=None, nullspace=None,
                 transpose_nullspace=None, near_nullspace=None, startup_params=None, **kwargs):

        super().__init__(F, t, dt, u0,
                         bcs=bcs, appctx=appctx, nullspace=nullspace)
        
        if isinstance(method, str):
            try:
                a, b = multistep_dict[method]
            except:
                raise ValueError(f'{method} is not a recognized method')
        else:
            a, b = method

        self.s = len(b) - 1
        self.a = vecconst(a)
        self.b = vecconst(b)
        self.us = []
        self.active_steps = []
        for i in range(len(self.a) - 1):
            if not (self.a[i] == zero() and self.b[i] == zero()):
                self.active_steps.append(i)
                self.us.append(u0.copy(deepcopy=True))
        self.us.append(u0)
        self.active_steps.append(len(self.a) - 1)
        
        Fnew, bcsnew = self.get_form_and_bcs(F, t, dt, u0, self.a, self.b, bcs=bcs)

        self.problem = NonlinearVariationalProblem(Fnew, self.us[-1], bcs=bcsnew, form_compiler_parameters=kwargs.pop("form_compiler_parameters", None),
            is_linear=kwargs.pop("is_linear", False),
            restrict=kwargs.pop("restrict", False))

        self.solver = NonlinearVariationalSolver(
            self.problem, appctx=self.appctx,
            nullspace=nullspace,
            transpose_nullspace=transpose_nullspace,
            near_nullspace=near_nullspace,
            solver_parameters=solver_parameters,
            **kwargs
            )

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        if bool(startup_params): 
            self.mechanized_startup(F, t, dt, u0, startup_params, bcs=bcs)

        self.bounds = bounds


    def get_form_and_bcs(self, F, t, dt, u0, a, b, bcs=None):

        F = expand_time_derivatives(F, t=t, timedep_coeffs=(u0, ))
        v, = F.arguments()
        V = v.function_space()

        assert V == u0.function_space()

        ## Is this the proper generalization?
        split_form = extract_terms(F)
        F_dt = split_form.time
        F_remainder = split_form.remainder

        # replace the time derivative with a linear combination of the previous steps
        temp_form = 0.0
        step_number = 0
        for (i, coeff) in enumerate(a):
            if (coeff is zero()) or (i not in self.active_steps):
                pass
            else:
                temp_form += coeff * self.us[self.active_steps.index(i)]
                step_number += 1

        dtu = TimeDerivative(u0)
        Fnew = replace(F_dt, {dtu: temp_form})

        # form the right hand side
        for (i, coeff) in enumerate(b):
            if (coeff is zero()) or (i not in self.active_steps):
                pass
            else:
                Fnew += dt * coeff * replace(F_remainder, {u0: self.us[self.active_steps.index(i)], 
                                                           t: t + (i - self.s + 1) * dt})
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


    # an optional method to mechanically find the required starting values via a single step method
    def mechanized_startup(self, F, t, dt, u0, startup_params, bcs=None):
        butcher_tableau = startup_params.get('tableau', None)
        stepper_kwargs = startup_params.get('stepper_kwargs', {})
        startup_dt_div = startup_params.get('dt_div', 1)
        assert isinstance(startup_dt_div, int) and startup_dt_div > 0

        self.us[0].assign(u0)
        self.TS = TimeStepper(F, butcher_tableau, t, dt, u0, bcs=bcs, **stepper_kwargs)

        
        # modify the timestep
        dt.assign(dt / startup_dt_div)
        
        # advance the system and assign values to previous steps
        counter = 1
        for i in range(self.s - 1):
            for substep in range(startup_dt_div):
                self.TS.advance()
                t.assign(t + dt)
            
            if self.a[i+1] is not zero():
                self.us[counter].assign(u0)
                counter += 1

        # reset the timestep
        dt.assign(dt * startup_dt_div)


    def solver_stats(self):

        return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)


valid_multistep_kwargs = ("bounds", "startup_params")


def MultistepTimeStepper(F, t, dt, u0, method, **kwargs):
    base_kwargs = {}
    for k in valid_base_kwargs:
        if k in kwargs:
            base_kwargs[k] = kwargs.pop(k)

    bcs = kwargs.pop("bcs", None)
    for cur_kwarg in kwargs.keys():
        if cur_kwarg not in valid_multistep_kwargs:
            raise ValueError(f"kwarg {cur_kwarg} is not allowable for MultistepTimeStepper")
        
    bounds = kwargs.pop('bounds', None)
    startup_params = kwargs.pop('startup_params', {})
    return MultistepStepper(F, t, dt, u0, method, bcs, startup_params=startup_params, bounds=bounds, **base_kwargs)
