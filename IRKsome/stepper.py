from .getForm import getForm
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake import norm, Function


class TimeStepper:
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None):
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = len(butcher_tableau.b)
        self.butcher_tableau = butcher_tableau

        bigF, stages, bigBCs, bigBCdata = \
            getForm(F, butcher_tableau, t, dt, u0, bcs)

        self.stages = stages
        self.bigBCs = bigBCs
        self.bigBCdata = bigBCdata
        problem = NLVP(bigF, stages, bigBCs)
        self.solver = NLVS(problem, solver_parameters=solver_parameters)

        self.ks = stages.split()

    def update(self):
        b = self.butcher_tableau.b
        dtc = float(self.dt)
        u0 = self.u0
        ns = self.num_stages
        nf = self.num_fields

        if nf == 1:
            ks = self.ks
            for i in range(ns):
                u0 += dtc * b[i] * ks[i]
        else:
            k = self.stages

            for s in range(ns):
                for i in range(nf):
                    u0.dat.data[i][:] += dtc * b[s] * k.dat.data[nf*s+i][:]

    def advance(self):
        for gdat, gcur in self.bigBCdata:
            gdat.interpolate(gcur)

        self.solver.solve()

        self.update()


class AdaptiveTimeStepper:
    def __init__(self, F, embedded_butcher_tableau, t, dt, u0,
                 tol=1.e-6, bcs=None, solver_parameters=None):
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = len(embedded_butcher_tableau.b)
        self.embedded_butcher_tableau = embedded_butcher_tableau
        self.delb = embedded_butcher_tableau.b - \
            embedded_butcher_tableau.btilde
        self.error_func = Function(u0.function_space())

    def advance(self):
        ord_m1 = self.embedded_butcher_tableau - 1

        err = 2.0 * self.tol

        while err >= self.tol:
            for gdat, gcur in self.bigBCdata:
                gdat.interpolate(gcur)

            self.solver.solve()

            err = self.estimate_error()

            q = 0.84 * (self.tol / err)**(ord_m1)
            q = max(min(q, 0.1), 4.0)
            dtnew = q * float(self.dt)

            if dtnew <= self.dt_min:
                raise RuntimeError("minimum time step encountered")
            else:
                self.dt.assign(dtnew)

        self.udpate()

    def estimate_error(self):
        dtc = float(self.dt)
        delb = self.delb

        if self.num_fields == 1:
            ks = self.ks
            self.error_func[:] = 0.0
            for i in range(self.num_stages):
                self.error_func += dtc * delb[i] * ks[i]
        else:
            k = self.stages
            for i in range(self.num_fields):
                self.error_func[i][:] = 0.0
            for s in range(self.num_stages):
                for i in range(self.num_fields):
                    self.error_func[i][:] += \
                        dtc * delb[s] * k.dat.data[self.num_fields*s+i][:]

        return norm(self.error_func)

    def update(self):
        b = self.butcher_tableau.b
        dtc = float(self.dt)
        u0 = self.u0
        nf = self.num_fields
        ns = self.num_stages

        if nf == 1:
            ks = self.ks
            for i in range(ns):
                u0 += dtc * b[i] * ks[i]
        else:
            k = self.stages
            for s in range(ns):
                for i in range(nf):
                    u0.dat.data[i][:] += dtc * b[s] * k.dat.data[nf*s+i][:]
