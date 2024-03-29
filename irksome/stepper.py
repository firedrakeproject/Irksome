import numpy
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake.dmhooks import pop_parent, push_parent
from .dirk_stepper import DIRKTimeStepper
from .getForm import AI, getForm
from .stage import StageValueTimeStepper
from .imex import RadauIIAIMEXMethod


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
    """

    valid_kwargs_per_stage_type = {
        "deriv": ["stage_type", "bcs", "nullspace", "solver_parameters", "appctx",
                  "bc_type", "splitting"],
        "value": ["stage_type", "bcs", "nullspace", "solver_parameters",
                  "update_solver_parameters", "appctx", "splitting"],
        "dirk": ["stage_type", "bcs", "nullspace", "solver_parameters", "appctx"],
        "imex": ["Fexp", "stage_type", "bcs", "nullspace",
                 "it_solver_parameters", "prop_solver_parameters",
                 "splitting", "appctx",
                 "num_its_initial", "num_its_per_step"]}

    stage_type = kwargs.get("stage_type", "deriv")
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
        return StageDerivativeTimeStepper(
            F, butcher_tableau, t, dt, u0, bcs, appctx=appctx,
            solver_parameters=solver_parameters, nullspace=nullspace,
            bc_type=bc_type, splitting=splitting)
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
