
def getFormGalerkin(F, V_trial, V_test, quad, t, dt, u0,
                    bcs = None, nullspace = None):

    """Given a time-dependent variational form, trial and test spaces, and
    a quadrature rule, produce UFL for the Galerkin-in-Time method.

    :arg F: UFL form for the semidiscrete ODE/DAE
    :arg V_trial: A :class:`firedrake.FunctionSpace` for the trial functions in time
    :arg V_test: A :class:`firedrake.FunctionSpace` for the test functions in time
    :arg quad: A :class:`fiat.QuadratureRule` for the time integration
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
         containing (possible time-dependent) boundary conditions imposed
         on the system.
    :arg nullspace: A list of tuples of the form (index, VSB) where
         index is an index into the function space associated with `u`
         and VSB is a :class: `firedrake.VectorSpaceBasis` instance to
         be passed to a `firedrake.MixedVectorSpaceBasis` over the
         larger space associated with the Runge-Kutta method

    On output, we return a tuple consisting of four parts:

       - Fnew, the :class:`Form`
       - bigBasis, the :class:`firedrake.Function` for the Galerkin solution.
         It lives in a :class:`firedrake.FunctionSpace` corresponding to the
         s-way tensor product of the space on which the semidiscrete
         form lives.
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the Galerkin-in-time solution,
       - 'nspnew', the :class:`firedrake.MixedVectorSpaceBasis` object
         that represents the nullspace of the coupled system
    """

    num_stages = # Infer from V_trial
    Vbig = reduce(mul, (V for _ in range(num_stages)))

    vnew = TestFunction(Vbig)
    w = Function(Vbig)



class GalkerinTimeStepper:
    """Front-end class for advancing a time-dependent PDE via a Galerkin
    in time method

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg V_trial: A :class:`firedrake.FunctionSpace` for the trial functions in time
    :arg V_test: A :class:`firedrake.FunctionSpace` for the test functions in time
    :arg quad: A :class:`fiat.QuadratureRule` for the time integration
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
            stage of the method.
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
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
    def __init__(self, F, V_trial, V_test, quad, t, dt, u0, bcs=None,
                 solver_parameters=None, appctx=None, nullspace=None):
        self.u0 = u0
        self.F = F
        self.orig_bcs = bcs
        self.t = t
        self.dt = dt
        self.V_trial = V_trial
        self.V_test = V_test
        self.quad = quad
        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        bigF, bigBasis, bigBCs, bigNSP = \
            getFormGalerkin(F, V_trial, V_test, quad, t, dt, u0, bcs, nullspace)

        self.bigBasis = bigBasis
        self.bigBCs = bigBCs
        problem = NLVP(bigF, bigBasis, bigBCs)
        appctx_irksome = {"F": F,
                          "t": t,
                          "dt": dt,
                          "u0": u0,
                          "bcs": bcs,
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

    def _update(self):
        """Assuming the algebraic problem for the Galerkin problems has been
        solved, updates the solution.  This will not typically be
        called by an end user."""

        ws = self.bigBasis
        u0bits = u0.subfunctions
        for s in range(ns):
            for i, u0bit in enumerate(u0bits):
                u0bit = 

    def advance(self):
        """Advances the system from time `t` to time `t + dt`.
        Note: overwrites the value `u0`."""
        push_parent(self.u0.function_space().dm, self.stages.function_space().dm)
        self.solver.solve()
        pop_parent(self.u0.function_space().dm, self.stages.function_space().dm)

        self.num_steps += 1
        self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()
        self._update()

    def solver_stats(self):
        return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)
