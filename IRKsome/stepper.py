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
    def __init__(self, F, butcher_tableau, t, dt, u0,
                 tol=1.e-6, dtmin=1.e-5, bcs=None, solver_parameters=None):
        assert butcher_tableau is not None
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.tol = tol
        self.dt_min = dtmin
        self.num_fields = len(u0.function_space())
        self.num_stages = len(butcher_tableau.b)
        self.butcher_tableau = butcher_tableau
        self.delb = butcher_tableau.b - \
            butcher_tableau.btilde
        self.error_func = Function(u0.function_space())

        bigF, stages, bigBCs, bigBCdata = \
            getForm(F, butcher_tableau, t, dt, u0, bcs)

        self.stages = stages
        self.bigBCs = bigBCs
        self.bigBCdata = bigBCdata
        problem = NLVP(bigF, stages, bigBCs)
        self.solver = NLVS(problem, solver_parameters=solver_parameters)

        self.ks = stages.split()

    def advance(self):
        ord_m1 = self.butcher_tableau.order - 1

        err = 2.0 * self.tol

        while err >= self.tol:
            print("\tTrying dt = ", float(self.dt))
            for gdat, gcur in self.bigBCdata:
                gdat.interpolate(gcur)

            self.solver.solve()

            err = self.estimate_error()
            print("\t truncation error: ", err)

            q = 0.84 * (self.tol / err)**(ord_m1)
            q = min(max(q, 0.1), 4.0)

            dtnew = q * float(self.dt)

            if dtnew <= self.dt_min:
                raise RuntimeError("minimum time step encountered")
            else:
                self.dt.assign(dtnew)
            if err < self.tol:
                print("\tSuccess")

        self.update()

    def estimate_error(self):
        dtc = float(self.dt)
        delb = self.delb

        if self.num_fields == 1:
            ks = self.ks
            self.error_func.dat.data[:] = 0.0
            for i in range(self.num_stages):
                self.error_func += dtc * delb[i] * ks[i]
        else:
            k = self.stages
            for i in range(self.num_fields):
                self.error_func.dat.data[i][:] = 0.0
            for s in range(self.num_stages):
                for i in range(self.num_fields):
                    self.error_func.dat.data[i][:] += \
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
