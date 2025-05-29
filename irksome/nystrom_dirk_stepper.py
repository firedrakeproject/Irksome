import numpy
from firedrake import (Function,
                       NonlinearVariationalProblem,
                       NonlinearVariationalSolver)
from ufl.constantvalue import as_ufl

from .deriv import Dt, expand_time_derivatives
from .tools import replace, MeshConstant, vecconst
from .bcs import bc2space
from .nystrom_stepper import butcher_to_nystrom, NystromTableau


def getFormDIRKNystrom(F, ks, tableau, t, dt, u0, ut0, bcs=None, bc_type=None):
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
            Dt(u0): g2 + k * (a * dt),
            Dt(u0, 2): k}
    stage_F = replace(F, repl)

    # BC's
    bcnew = []

    # For the DIRK case, we need one new BC for each old one (rather
    # than one per stage), but we need a `Function` inside of each BC
    # and a rule for computing that function at each time for each
    # stage.
    abar_vals = numpy.array([MC.Constant(0) for i in range(num_stages)],
                            dtype=object)
    d_val = MC.Constant(1.0)
    if bc_type == "DAE":
        # Here, at each stage, abar_vals should include the
        # subdiagonal values from Abar in the Nystrom Tableau and
        # d_val should include the diagonal value from Abar
        for bc in bcs:
            bcarg = bc._original_arg
            if bcarg == 0:
                # Homogeneous BC, just zero out stage dofs
                bcnew.append(bc)
            else:
                bcarg_stage = replace(as_ufl(bcarg), {t: t+c*dt})
                gdat = bcarg_stage - bc2space(bc, u0) - c*dt*bc2space(bc, ut0)
                gdat -= sum(bc2space(bc, ks[i]) * (abar_vals[i] * dt**2) for i in range(num_stages))
                gdat /= d_val * dt**2
                bcnew.append(bc.reconstruct(g=gdat))
    elif bc_type == "dDAE":
        # Here, at each implicit stage, abar_vals should include the
        # subdiagonal values from A in the Nystrom Tableau and
        # d_val should include the diagonal value from A
        # If explicit, we omit the first stage from this calculation,
        # and augment with the final time reconstruction; this is
        # handled by AABbar and CCone in the later code
        for bc in bcs:
            bcarg = bc._original_arg
            if bcarg == 0:
                # Homogeneous BC, just zero out stage dofs
                bcnew.append(bc)
            else:
                bcprime = expand_time_derivatives(Dt(as_ufl(bcarg)), t=t, timedep_coefs=(u0,))
                bcprime_stage = replace(bcprime, {t: t+c*dt})
                gdat = bcprime_stage - bc2space(bc, ut0)
                gdat -= sum(bc2space(bc, ks[i]) * (abar_vals[i] * dt) for i in range(num_stages))
                gdat /= d_val * dt
                bcnew.append(bc.reconstruct(g=gdat))

    return stage_F, (k, g1, g2, a, abar, c), bcnew, (abar_vals, d_val)


class DIRKNystromTimeStepper:
    """Front-end class for advancing a second-order time-dependent PDE via a diagonally-implicit
    Runge-Kutta-Nystrom method formulated in terms of stage derivatives."""

    def __init__(self, F, tableau, t, dt, u0, ut0, bcs=None,
                 solver_parameters=None,
                 appctx=None, nullspace=None,
                 transpose_nullspace=None, near_nullspace=None,
                 bc_type=None,
                 **kwargs):
        if not isinstance(tableau, NystromTableau):
            tableau = butcher_to_nystrom(tableau)
        assert tableau.is_diagonally_implicit
        if bc_type is None:
            bc_type = "DAE"

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        self.tableau = tableau
        self.num_stages = num_stages = tableau.num_stages

        self.AA = vecconst(tableau.A)
        self.AAbar = vecconst(tableau.Abar)
        self.BB = vecconst(tableau.b)
        self.BBbar = vecconst(tableau.bbar)
        self.CC = vecconst(tableau.c)

        if bc_type == "DAE":
            if tableau.is_explicit:
                raise NotImplementedError("Cannot have DAE BCs with Explicit Nystrom methods")
            self.AABbar = vecconst(tableau.Abar)
            self.CCone = vecconst(tableau.c)
        elif bc_type == "dDAE":
            if tableau.is_explicit:
                AABbar = numpy.vstack((tableau.A, tableau.b))
                self.AABbar = vecconst(AABbar[1:])
                CCone = numpy.append(tableau.c[1:], 1.0)
                self.CCone = vecconst(CCone)
            else:
                self.AABbar = vecconst(tableau.A)
                self.CCone = vecconst(tableau.c)
        else:
            raise NotImplementedError(f"No implementation for bc_type {bc_type} for DIRK-Nystrom or Explicit-Nystrom methods")

        self.V = V = u0.function_space()
        self.u0 = u0
        self.ut0 = ut0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.ks = [Function(V) for _ in range(num_stages)]

        stage_F, self.kgac, bcnew, (abar_vals, d_val) = getFormDIRKNystrom(
            F, self.ks, tableau, t, dt, u0, ut0, bcs=bcs)

        k = self.kgac[0]
        self.bcnew = bcnew

        appctx_irksome = {"stepper": self}
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

        self.bc_constants = abar_vals, d_val

    def update_bc_constants(self, i, c):
        AAbar = self.AABbar
        CCone = self.CCone
        abar_vals, d_val = self.bc_constants
        ns = AAbar.shape[1]
        for j in range(i):
            abar_vals[j].assign(AAbar[i, j])
        for j in range(i, ns):
            abar_vals[j].assign(0)
        d_val.assign(AAbar[i, i])
        c.assign(CCone[i])

    def advance(self):
        k, g1, g2, a, abar, c = self.kgac
        ks = self.ks
        u0 = self.u0
        ut0 = self.ut0
        dt = self.dt
        for i in range(self.num_stages):
            g1.assign(sum((ks[j] * (self.AAbar[i, j] * dt**2) for j in range(i)),
                          u0 + ut0 * (self.CC[i] * dt)))
            g2.assign(sum((ks[j] * (self.AA[i, j] * dt) for j in range(i)), ut0))
            self.update_bc_constants(i, c)
            a.assign(self.AA[i, i])
            abar.assign(self.AAbar[i, i])
            self.solver.solve()
            self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
            self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()
            ks[i].assign(k)

        # update the solution with now-computed stage values.
        u0 += ut0 * dt + sum(ks[i] * (self.BBbar[i] * dt**2) for i in range(self.num_stages))
        ut0 += sum(ks[i] * (self.BB[i] * dt) for i in range(self.num_stages))

        self.num_steps += 1

    def solver_stats(self):
        return self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations


class ExplicitNystromTimeStepper(DIRKNystromTimeStepper):
    """Front-end class for advancing a second-order time-dependent PDE via an explicit
    Runge-Kutta-Nystrom method formulated in terms of stage derivatives."""

    def __init__(self, F, tableau, t, dt, u0, ut0, bcs=None,
                 solver_parameters=None,
                 appctx=None, nullspace=None,
                 transpose_nullspace=None, near_nullspace=None,
                 bc_type=None,
                 **kwargs):
        if not isinstance(tableau, NystromTableau):
            tableau = butcher_to_nystrom(tableau)
        assert tableau.is_explicit
        if bc_type is None:
            bc_type = "dDAE"

        # we just have one mass matrix we're reusing for each time step and
        # each stage, so we can nudge this along
        solver_parameters = {} if solver_parameters is None else solver_parameters
        solver_parameters.setdefault("snes_lag_jacobian_persists", True)
        solver_parameters.setdefault("snes_lag_jacobian", -2)
        solver_parameters.setdefault("snes_lag_preconditioner_persists", True)
        solver_parameters.setdefault("snes_lag_preconditioner", -2)
        super(ExplicitNystromTimeStepper, self).__init__(
            F, tableau, t, dt, u0, ut0, bcs=bcs,
            solver_parameters=solver_parameters, appctx=appctx,
            nullspace=None,
            transpose_nullspace=None, near_nullspace=None,
            bc_type=None,
            **kwargs)
