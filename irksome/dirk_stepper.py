import numpy
from firedrake import Function
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from ufl.constantvalue import as_ufl

from .deriv import TimeDerivative
from .tools import component_replace, replace, MeshConstant, vecconst
from .bcs import bc2space


def getFormDIRK(F, ks, butch, t, dt, u0, bcs=None):
    if bcs is None:
        bcs = []

    v = F.arguments()[0]
    V = v.function_space()
    msh = V.mesh()
    assert V == u0.function_space()

    num_stages = butch.num_stages
    k = Function(V)
    g = Function(V)

    # Note: the Constant c is used for substitution in both the
    # variational form and BC's, and we update it for each stage in
    # the loop over stages in the advance method.  The Constant a is
    # used similarly in the variational form
    MC = MeshConstant(msh)
    c = MC.Constant(1.0)
    a = MC.Constant(1.0)

    repl = {t: t + c * dt,
            u0: g + k * (a * dt),
            TimeDerivative(u0): k}
    stage_F = component_replace(F, repl)

    bcnew = []

    # For the DIRK case, we need one new BC for each old one (rather
    # than one per stage), but we need a `Function` inside of each BC
    # and a rule for computing that function at each time for each
    # stage.
    a_vals = numpy.array([MC.Constant(0) for i in range(num_stages)],
                         dtype=object)
    d_val = MC.Constant(1.0)
    for bc in bcs:
        bcarg = bc._original_arg
        if bcarg == 0:
            # Homogeneous BC, just zero out stage dofs
            bcnew.append(bc)
        else:
            bcarg_stage = replace(as_ufl(bcarg), {t: t+c*dt})
            gdat = bcarg_stage - bc2space(bc, u0)
            gdat -= sum(bc2space(bc, ks[i]) * (a_vals[i] * dt) for i in range(num_stages))
            gdat /= d_val * dt
            bcnew.append(bc.reconstruct(g=gdat))

    return stage_F, (k, g, a, c), bcnew, (a_vals, d_val)


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
        self.num_stages = num_stages = butcher_tableau.num_stages
        self.AAb = numpy.vstack((butcher_tableau.A, butcher_tableau.b))
        self.CCone = numpy.append(butcher_tableau.c, 1.0)

        # Need to be able to set BCs for either the DIRK or explicit cases.

        # For DIRK, we say that the stage i solution should match the
        # prescribed boundary values at time c[i], which means we use
        # the i^th row of the Butcher tableau in determining the
        # boundary condition

        # For explicit, we say that the stage i solution should be
        # determined to match the prescribed boundary values at time
        # c[i+1] (the first stage where it appears), which means we
        # use the (i+1)^st row of the Butcher tableau in determining
        # the boundary condition, and the full reconstruction for the
        # final stage

        if butcher_tableau.is_explicit:
            self.AAb = self.AAb[1:]
            self.CCone = self.CCone[1:]
        self.AA = vecconst(butcher_tableau.A)
        self.BB = vecconst(butcher_tableau.b)

        self.V = V = u0.function_space()
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.ks = [Function(V) for _ in range(num_stages)]

        # "k" is a generic function for which we will solve the
        # NVLP for the next stage value
        # "ks" is a list of functions for the stage values
        # that we update as we go.  We need to remember the
        # stage values we've computed earlier in the time step...

        stage_F, (k, g, a, c), bcnew, (a_vals, d_val) = getFormDIRK(
            F, self.ks, butcher_tableau, t, dt, u0, bcs=bcs)

        self.bcnew = bcnew

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
        self.bc_constants = a_vals, d_val

    def update_bc_constants(self, i, c):
        AAb = self.AAb
        CCone = self.CCone
        a_vals, d_val = self.bc_constants
        ns = AAb.shape[1]
        for j in range(i):
            a_vals[j].assign(AAb[i, j])
        for j in range(i, ns):
            a_vals[j].assign(0)
        d_val.assign(AAb[i, i])
        c.assign(CCone[i])

    def advance(self):
        k, g, a, c = self.kgac
        ks = self.ks
        u0 = self.u0
        dt = self.dt
        for i in range(self.num_stages):
            # compute the already-known part of the state in the
            # variational form
            g.assign(sum((ks[j] * (self.AA[i, j] * dt) for j in range(i)), u0))

            # update BC constants for the variational problem
            self.update_bc_constants(i, c)
            a.assign(self.AA[i, i])

            # solve new variational problem, stash the computed
            # stage value.

            # Note: implicitly uses solution value for stage i as
            # initial guess for stage i+1 and uses the last stage from
            # previous time step for stage 0 of the next one.  The
            # former is probably optimal, we hope for the best with
            # the latter.
            self.solver.solve()
            self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
            self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()
            ks[i].assign(k)

        # update the solution with now-computed stage values.
        u0 += sum(ks[i] * (self.BB[i] * dt) for i in range(self.num_stages))

        self.num_steps += 1

    def solver_stats(self):
        return self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations
