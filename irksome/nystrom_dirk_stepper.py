import numpy
from firedrake import (Function,
                       NonlinearVariationalProblem,
                       NonlinearVariationalSolver)
from ufl.constantvalue import as_ufl

from .deriv import TimeDerivative, expand_time_derivatives
from .tools import replace, MeshConstant, vecconst
from .bcs import bc2space


def getFormNystromDIRK(F, ks, tableau, t, dt, u0, ut0, bcs=None, bc_type=None):
    if bcs is None:
        bcs = []
    if bc_type is None:
        bc_type = "DAE"

    v = F.arguments()[0]
    V = v.function_space()
    msh = V.mesh()
    assert V == u0.function_space()

    num_stages = tableau.num_stages
    k = Function(V)
    g1 = Function(V)
    g2 = Function(V)

    # Note: the Constant c is used for substitution in both the
    # variational form and BC's, and we update it for each stage in
    # the loop over stages in the advance method.  The Constants a
    # and abar are used similarly in the variational form
    MC = MeshConstant(msh)
    c = MC.Constant(1.0)
    a = MC.Constant(1.0)
    abar = MC.Constant(1.0)

    # preprocess time derivatives
    F = expand_time_derivatives(F, t=t, timedep_coeffs=(u0,))

    repl = {t: t + c * dt,
            u0: g1 + k * (abar * dt**2),
            TimeDerivative(u0): g2 + k * (a * dt),
            TimeDerivative(TimeDerivative(u0)): k}
    stage_F = replace(F, repl)

    # BC's (TODO!)

    return stage_F, (k, g1, g2, a, abar, c), None, None


class NystromDIRKTimeStepper:
    """Front-end class for advancing a second-order time-dependent PDE via a diagonally-implicit
    Runge-Kutta-Nystrom method formulated in terms of stage derivatives."""

    def __init__(self, F, tableau, t, dt, u0, ut0, bcs=None,
                 solver_parameters=None,
                 appctx=None, nullspace=None,
                 transpose_nullspace=None, near_nullspace=None,
                 bc_type=None,
                 **kwargs):
        assert tableau.is_diagonally_implicit
        if bc_type is None:
            bc_type = "DAE"

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        self.tableau = tableau
        self.num_stages = num_stages = tableau.num_stages

        self.AA = vecconst(tableau.A)
        self.AAbar = vecconst(tableau.A)
        self.BB = vecconst(tableau.b)
        self.BBbar = vecconst(tableau.bbar)
        self.CC = vecconst(tableau.c)

        self.V = V = u0.function_space()
        self.u0 = u0
        self.ut0 = ut0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.ks = [Function(V) for _ in range(num_stages)]

        stage_F, self.kgac, bcnew, _ = getFormNystromDIRK(
            F, self.ks, tableau, t, dt, u0, ut0, bcs=bcs)

        self.bcnew = bcnew

        appctx_irksome = {"F": F,
                          "stage_type": "dirk",
                          "nystrom_tableau": tableau,
                          "t": t,
                          "dt": dt,
                          "u0": u0,
                          "ut0": ut0,
                          "bcs": bcs,
                          "bc_type": bc_type,
                          "nullspace": nullspace}
        if appctx is None:
            appctx = appctx_irksome
        else:
            appctx = {**appctx, **appctx_irksome}
        self.appctx = appctx

        self.problem = NonlinearVariationalProblem(
            stage_F, k, bcs=bcnew,
            form_compiler_parameters=kwargs.pop("form_compiler_parameters", None),
            is_linear=kwargs.pop("is_linear", False),
            restrict=kwargs.pop("restrict", False),
        )
        self.solver = NonlinearVariationalSolver(
            self.problem, appctx=appctx,
            nullspace=nullspace,
            transpose_nullspace=transpose_nullspace,
            near_nullspace=near_nullspace,
            solver_parameters=solver_parameters,
            **kwargs,
        )

        self.bc_constants = None

    def update_bc_constants(self, i, c):
        pass

    def advance(self):
        k, g1, g2, a, abar, c = self.kgac
        ks = self.ks
        u0 = self.u0
        ut0 = self.ut0
        dt = self.dt
        for i in range(self.num_stages):
            g1.assign(sum((ks[j] * (self.AAbar[i, j] * dt**2) for j in range(i)),
                          u0, ut0 * (self.CC[i] * dt)))
            g2.assign(sum((ks[j] * (self.AA[i, j] * dt) for j in range(i)), ut0))
            self.update_bc_constants(i, c)
            a.assign(self.AA[i, i])
            abar.assign(self.AAbar[i, i])
            self.solver.solve()
            self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
            self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()
            ks[i].assign(k)

        # update the solution with now-computed stage values.
        u0 += ut0 * dt + sum(ks[i] * (self.BBbar[i] * dt) for i in range(self.num_stages))
        ut0 += sum(ks[i] * (self.BB[i] * dt) for i in range(self.num_stages))

        self.num_steps += 1

    def solver_stats(self):
        return self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations
