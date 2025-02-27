from abc import abstractmethod


class BaseTimeStepper:
    def __init__(self, F, t, dt, u0,
                 bcs=None, appctx=None, nullspace=None):
        self.F = F
        self.t = t
        self.dt = dt
        self.u0 = u0
        self.bcs = bcs
        self.nullspace = nullspace

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

        super().__init__(F, butcher_tableau, t, dt, u0,
                         bcs=bcs, appctx=appctx, nullspace=nullspace)


    def advance(self):
        self.solver.solve()
        self._update()

    # allow butcher tableau as input for preconditioners to create
    # an alternate operator
    @abstractmethod
    def getForm(self, butch=None):
        pass


