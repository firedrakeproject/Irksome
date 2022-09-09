from .getForm import getForm, AI
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake.dmhooks import pop_parent, push_parent
import numpy
from .stage import StageValueTimeStepper
from .dirk_stepper import DIRKTimeStepper


def TimeStepper(F, butcher_tableau, t, dt, u0, bcs=None,
                solver_parameters=None,
                update_solver_parameters=None,
                nullspace=None,
                stage_type="deriv", appctx=None,
                bc_type=None, splitting=None):
    """Helper function to dispatch between various back-end classes
       for doing time stepping.  Returns an instance of the
       appropriate class.

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` iss the
            :class:firedrake.TestFunction`.
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
            the Runge-Kutta method to be used for time marching.
    :arg t: A :class:`firedrake.Constant` instance that always
            contains the time value at the beginning of a time step
    :arg dt: A :class:`firedrake.Constant` containing the size of the
            current time step.  The user may adjust this value between
            time steps.
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
    if stage_type == "deriv":
        if bc_type is None:
            bc_type = "DAE"
        if splitting is None:
            splitting = AI
        assert update_solver_parameters is None
        return StageDerivativeTimeStepper(
            F, butcher_tableau, t, dt, u0, bcs, appctx=appctx,
            solver_parameters=solver_parameters, nullspace=nullspace,
            bc_type=bc_type, splitting=splitting)
    elif stage_type == "value":
        assert bc_type is None
        if splitting is None:
            splitting = AI
        return StageValueTimeStepper(
            F, butcher_tableau, t, dt, u0, bcs=bcs, appctx=appctx,
            solver_parameters=solver_parameters,
            splitting=splitting,
            update_solver_parameters=update_solver_parameters,
            nullspace=nullspace)
    elif stage_type == "dirk":
        assert bc_type is None and splitting is None \
            and update_solver_parameters is None
        return DIRKTimeStepper(
            F, butcher_tableau, t, dt, u0, bcs,
            solver_parameters, appctx, nullspace)


class StageDerivativeTimeStepper:
    """Front-end class for advancing a time-dependent PDE via a Runge-Kutta
    method formulated in terms of stage derivatives.

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
            time steps.
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
            self.ws = stages.split()

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
        for s in range(ns):
            for i, u0d in enumerate(u0.dat):
                u0d.data[:] += dtc * b[s] * ws[nf*s+i].dat.data_ro

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
        for i, u0d in enumerate(u0.dat):
            u0d.data[:] += dtc * ws[nf*(ns-1)+i].dat.data_ro

    def advance(self):
        """Advances the system from time `t` to time `t + dt`.
        Note: overwrites the value `u0`."""
        for gdat, gcur, gmethod in self.bigBCdata:
            gmethod(gcur, self.u0)

        push_parent(self.u0.function_space().dm, self.stages.function_space().dm)
        self.solver.solve()
        pop_parent(self.u0.function_space().dm, self.stages.function_space().dm)

        self._update()
