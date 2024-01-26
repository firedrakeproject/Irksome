import numpy
from firedrake import DirichletBC, Function
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake import interpolate, split
from ufl.constantvalue import as_ufl

from .deriv import TimeDerivative
from .tools import replace, MeshConstant


class BCThingy:
    def __init__(self):
        pass

    def __call__(self, u):
        return u


class BCCompOfNotMixedThingy:
    def __init__(self, comp):
        self.comp = comp

    def __call__(self, u):
        return u[self.comp]


class BCMixedBitThingy:
    def __init__(self, sub):
        self.sub = sub

    def __call__(self, u):
        return u.sub(self.sub)


class BCCompOfMixedBitThingy:
    def __init__(self, sub, comp):
        self.sub = sub
        self.comp = comp

    def __call__(self, u):
        return u.sub(self.sub)[self.comp]


def getThingy(V, bc):
    num_fields = len(V)
    Vbc = bc.function_space()
    if num_fields == 1:
        comp = Vbc.component
        if comp is None:
            return BCThingy()
        else:
            return BCCompOfNotMixedThingy(comp)
    else:
        sub = bc.function_space_index()
        comp = Vbc.component
        if comp is None:
            return BCMixedBitThingy(sub)
        else:
            return BCCompOfMixedBitThingy(sub, comp)


def getFormDIRK(F, butch, t, dt, u0, bcs=None):
    if bcs is None:
        bcs = []

    v = F.arguments()[0]
    V = v.function_space()
    msh = V.mesh()
    assert V == u0.function_space()

    num_fields = len(V)

    k = Function(V)
    g = Function(V)

    # If we're on a mixed problem, we need to replace pieces of the
    # solution.  Stores an array of the splittings of the k for each stage.
    if num_fields == 1:
        k_bits = [k]
        u0bits = [u0]
        gbits = [g]
    else:
        k_bits = numpy.array(split(k), dtype=object)
        u0bits = split(u0)
        gbits = split(g)

    # Note: the Constant c is used for substitution in both the
    # variational form and BC's, and we update it for each stage in
    # the loop over stages in the advance method.  The Constant a is
    # used similarly in the variational form
    MC = MeshConstant(msh)
    c = MC.Constant(1.0)
    a = MC.Constant(1.0)

    repl = {t: t+c*dt}
    for u0bit, kbit, gbit in zip(u0bits, k_bits, gbits):
        repl[u0bit] = gbit + dt * a * kbit
        repl[TimeDerivative(u0bit)] = kbit
    stage_F = replace(F, repl)

    bcnew = []
    gblah = []

    # For the DIRK case, we need one new BC for each old one (rather
    # than one per stage), but we need a `Function` inside of each BC
    # and a rule for computing that function at each time for each
    # stage.
    for bc in bcs:
        Vbc = bc.function_space()
        bcarg = as_ufl(bc._original_arg)
        bcarg_stage = replace(bcarg, {t: t+c*dt})
        try:
            gdat = interpolate(bcarg, Vbc)
            gmethod = lambda gd, gc: gd.interpolate(gc)
        except:  # noqa: E722
            gdat = interpolate(bcarg, Vbc)
            gmethod = lambda gd, gc: gd.project(gc)

        new_bc = DirichletBC(Vbc, gdat, bc.sub_domain)
        bcnew.append(new_bc)

        dat4bc = getThingy(V, bc)
        gdat2 = Function(gdat.function_space())
        gblah.append((gdat, gdat2, bcarg_stage, gmethod, dat4bc))

    return stage_F, (k, g, a, c), bcnew, gblah


class DIRKTimeStepper:
    """Front-end class for advancing a time-dependent PDE via a diagonally-implicit
    Runge-Kutta method formulated in terms of stage derivatives."""

    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None,
                 appctx=None, nullspace=None):
        assert butcher_tableau.is_diagonally_implicit

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        self.butcher_tableau = butcher_tableau
        self.V = V = u0.function_space()
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = num_stages = butcher_tableau.num_stages
        self.ks = [Function(V) for _ in range(num_stages)]

        # "k" is a generic function for which we will solve the
        # NVLP for the next stage value
        # "ks" is a list of functions for the stage values
        # that we update as we go.  We need to remember the
        # stage values we've computed earlier in the time step...

        stage_F, (k, g, a, c), bcnew, gblah = getFormDIRK(
            F, butcher_tableau, t, dt, u0, bcs=bcs)

        self.bcnew = bcnew
        self.gblah = gblah

        appctx_irksome = {"F": F,
                          "butcher_tableau": butcher_tableau,
                          "t": t,
                          "dt": dt,
                          "u0": u0,
                          "bcs": bcs,
                          "bc_type": "DAE",
                          "nullspace": nullspace}
        if appctx is None:
            appctx = appctx_irksome
        else:
            appctx = {**appctx, **appctx_irksome}

        self.problem = NLVP(stage_F, k, bcnew)
        self.solver = NLVS(
            self.problem, appctx=appctx, solver_parameters=solver_parameters,
            nullspace=nullspace)

        self.kgac = k, g, a, c

    def advance(self):
        k, g, a, c = self.kgac
        ks = self.ks
        u0 = self.u0
        dtc = float(self.dt)
        bt = self.butcher_tableau
        AA = bt.A
        CC = bt.c
        BB = bt.b
        gsplit = g.subfunctions
        for i in range(self.num_stages):
            # update a, c constants tucked into the variational problem
            # for the current stage
            a.assign(AA[i, i])
            c.assign(CC[i])
            # compute the already-known part of the state in the
            # variational form
            g.assign(u0)
            for j in range(i):
                ksplit = ks[j].subfunctions
                for (gbit, kbit) in zip(gsplit, ksplit):
                    gbit += dtc * float(AA[i, j]) * kbit

            # update BC's for the variational problem
            for (bc, (gdat, gdat2, gcur, gmethod, dat4bc)) in zip(self.bcnew, self.gblah):
                # Evaluate the Dirichlet BC at the current stage time
                gmethod(gdat, gcur)

                gmethod(gdat2, dat4bc(u0))
                gdat -= gdat2

                # Subtract previous stage values
                for j in range(i):
                    gmethod(gdat2, dat4bc(ks[j]))
                    gdat -= dtc * float(AA[i, j]) * gdat2

                # Rescale gdat
                gdat /= dtc * float(AA[i, i])

            # solve new variational problem, stash the computed
            # stage value.

            # Note: implicitly uses solution value for stage i as
            # initial guess for stage i+1 and uses the last stage from
            # previous time step for stage 0 of the next one.  The
            # former is probably optimal, we hope for the best with
            # the latter.
            self.solver.solve()
            mysnes = self.solver.snes
            self.num_nonlinear_iterations += mysnes.getIterationNumber()
            self.num_linear_iterations += mysnes.getLinearSolveIterations()
            ks[i].assign(k)

        # update the solution with now-computed stage values.
        for i in range(self.num_stages):
            for (u0bit, kbit) in zip(u0.subfunctions, ks[i].subfunctions):
                u0bit += dtc * float(BB[i]) * kbit

        self.num_steps += 1

    def solver_stats(self):
        return self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations
