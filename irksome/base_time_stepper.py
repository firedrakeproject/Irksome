from abc import abstractmethod
from functools import reduce
from operator import mul
from firedrake import Function


class BaseTimeStepper:
    def __init__(self, F, t, dt, u0,
                 bcs=None, appctx=None, nullspace=None):
        self.F = F
        self.t = t
        self.dt = dt
        self.u0 = u0
        self.orig_bcs = bcs
        self.nullspace = nullspace
        self.V = u0.function_space()

        appctx_base = {
            "F": F,
            "t": t,
            "dt": dt,
            "u0": u0,
            "bcs": bcs,
            "nullspace": nullspace}

        if appctx is None:
            self.appctx = appctx_base
        else:
            self.appctx = {**appctx, **appctx_base}

    @abstractmethod
    def advance(self):
        pass

    @abstractmethod
    def solver_stats(self):
        pass


# Stage derivative + stage value + (maybe?) RadauIIAIMEX
class StageCoupledTimeStepper(BaseTimeStepper):
    def __init__(self, F, t, dt, u0,
                 bcs=None, solver_parameters=None,
                 appctx=None, nullspace=None,
                 splitting=None, bc_type="DAE"):

        super().__init__(F, t, dt, u0,
                         bcs=bcs, appctx=appctx, nullspace=nullspace)
        self.splitting = splitting
        self.bc_type = bc_type

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

    def advance(self):
        self.solver.solve()
        self._update()

    # allow butcher tableau as input for preconditioners to create
    # an alternate operator
    @abstractmethod
    def get_form_and_bcs(self, stages, butcher_tableau=None):
        pass

    def solver_stats(self):
        return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)

    def get_stages(self):
        num_stages = self.butcher_tableau.num_stages
        Vbig = reduce(mul, (self.V for _ in range(num_stages)))
        return Function(Vbig)
