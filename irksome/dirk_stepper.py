from .getForm import getForm, AI
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake.dmhooks import pop_parent, push_parent
import numpy
from ufl.classes import Zero
from .tools import replace, vecconst


def getFormDIRK(F, butch, t, dt, u0, bcs=None):
    num_stages = butch.num_stages

    v = F.arguments()[0]
    V = v.function_space()

    num_fields = len(V)

    k = Function(V)
    g = Function(V)
    
    # If we're on a mixed problem, we need to replace pieces of the
    # solution.  Stores an array of the splittings of the k for each stage.
    k_bits = numpy.array(split(k))

    u0bits = split(u0)
    vbits = split(v)
    gbits = split(g)
    
    c = Constant(1.0)
    a = Constant(1.0)

    repl = {t: t+c*dt}
    for ubit, kbit, gbit in zip(u0bits, kbit, gbits))):
        repl[u0bit] = gbit + dt * a * k_bit
        repl[TimeDerivative(u0bit)] = kbit
    stage_F = replace(F, repl)
        
    # TODO: BCs!

    return stage_Fs, (k, g, a, c), None, None


class DIRKTimeStepper:
    """Front-end class for advancing a time-dependent PDE via a diagonally-implicit 
    Runge-Kutta method formulated in terms of stage derivatives."""

    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None,
                 appctx=None, nullspace=None):
        assert butcher_tableau.is_diagonally_implicit
        self.butcher_tableau = butcher_tableau
        self.V = V = u0.function_space()
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = num_stages = butcher_tableau.num_stages
        self.ks = [Function(V) for _ in range(num_stages)]

        # "k" is a generic function that we will solve the
        # NVLP for the next stage value
        # "ks" is a list of functions for the stage values
        # that we update as we go.  We need to remember the
        # stage values we've computed earlier in the time step...
        
        stage_F, (k, g, a, c), _, _ = getFormDIRK(
            F, butch, t, dt, u0, bcs=bcs)

        appctx_irksome = {"F": F,
                          "butcher_tableau": butcher_tableau,
                          "t": t,
                          "dt": dt,
                          "u0": u0,
                          "bcs": bcs,
                          "bc_type": "DAE",
                          "nullspace": nullspace}

        self.problem = NLVP(stage_F, k)
        self.solver = NLVS(
            problem, appctx=appctx, solver_paramters=solver_parameters,
            nullspace=nullspace)

        self.kgac = k, g, a, c
        
    def advance(self):
        k, g, a, c = self.kgac
        ks = self.ks
        u0 = self.u0
        dtc = float(self.dt)
        bt = self.butcher_tableau
        AA = bt.A
        CC = bt.C
        BB = bt.B
        for i in range(self.num_stages):
            # update a, c constants tucked into the variational problem
            # for the current stage
            a.assign(AA[i, i])
            c.assign(CC[i])
            # compute the already-known part of the state in the
            # variational form
            g.assign(u0)
            for j in range(i):
                for (gd, kd) in zip(g.dat, ks[j].dat):
                    g.data[:] += dtc * AA[i, j] * kd.data_ro[:]
            # solve new variational problem, stash the computed
            # stage value.
            # Note: implicitly uses solution value for
            # stage i as initial guess for stage i+1
            # and uses the last stage from previous time step
            # for stage 0 of the next one.
            self.solver.solve()
            ks[i].assign(k)

        # update the solution with now-computed stage values.
        for i in range(self.num_stages):
            for (u0d, kd) in zip(u0.dat, ks[i].dat):
                u0d.data[:] += dtc * BB[i] * kd.data_ro[:]
    
        
