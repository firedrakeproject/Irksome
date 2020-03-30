from .getForm import getForm
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS


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

        if self.num_fields == 1:
            self.update_solution = self._update_simple
        else:
            self.update_solution = self._update_mixed
            
    def advance(self):
        for gdat, gcur in self.bigBCdata:
            gdat.interpolate(gcur)

        self.solver.solve()
        self.update_solution()

        self.t.assign(self.t.values()[0] + self.dt.values()[0])


    def _update_simple(self):
        b = self.butcher_tableau.b
        ks = self.ks
        dtc = self.dt.values()[0]
        for i in range(self.num_stages):
            self.u0 += dtc * b[i] * ks[i]


    def _update_mixed(self):
        b = self.butcher_tableau.b
        u0 = self.u0
        dtc = self.dt.values()[0]
        k = self.stages

        for s in range(self.num_stages):
            for i in range(self.num_fields):
                u0.dat.data[i][:] += dtc * k.dat.data[num_fields*s+i][:]
