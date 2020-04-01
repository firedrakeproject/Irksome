from .getForm import getForm
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake import norm
import numpy


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


    def advance(self):
        for gdat, gcur in self.bigBCdata:
            gdat.interpolate(gcur)

        self.solver.solve()
        
        b = self.butcher_tableau.b
        dtc = float(self.dt)
        u0 = self.u0

        if self.num_fields == 1:
            ks = self.ks
            for i in range(self.num_stages):
                u0 += dtc * b[i] * ks[i]
        else:
            k = self.stages
            for s in range(self.num_stages):
                for i in range(num_fields):
                    u0.dat.data[i][:] += dtc * b[s] * k.dat.data[num_fields*s+i][:]


class AdaptiveTimeStepper:
    def __init__(self, F, embedded_butcher_tableau, t, dt, u0, tol=1.e-6, bcs=None,
                 solver_parameters=None):
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = len(butcher_tableau.b)
        self.embedded_butcher_tableau = embedded_butcher_tableau
        self.delb = embedded_butcher_tableau.b - embedded_butcher_tableau.btilde
        self.error_func = Function(u0.function_space())
        
    def advance(self):
        accept_step = False

        while float(self.dt) >= self.dt_min and not accept_step:
            for gdat, gcur in self.bigBCdata:
                gdat.interpolate(gcur)

            self.solver.solve()

            err = self.estimate_error()

            if err <= self.tol:
                dtnew 

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
                for i in range(num_fields):
                    self.error_func[i][:] += dtc * delb[s] * k.dat.data[self.num_fields*s+i][:]
                
        return norm(self.error_func)
        
        
