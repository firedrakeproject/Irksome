import numpy
from ufl import as_ufl, lhs

from .ufl.deriv import Dt, expand_time_derivatives
from .tools import extract_timedep_arguments, replace
from .constant import MeshConstant, vecconst
from .nystrom_stepper import butcher_to_nystrom, NystromTableau
from .backend import get_backend


def getFormDIRKNystrom(F, ks, tableau, t, dt, u0, ut0, bcs=None, bc_type=None, kgac=None, backend="firedrake"):
    backend_cls = get_backend(backend)
    if bcs is None:
        bcs = []
    if bc_type is None:
        bc_type = "DAE"

    v, u = extract_timedep_arguments(F, u0)
    V = backend_cls.get_function_space(v)
    assert V == backend_cls.get_function_space(u0)

    # preprocess time derivatives
    F = expand_time_derivatives(F, t=t, timedep_coeffs=(u,))

    num_stages = tableau.num_stages

    # Note: the Constant c is used for substitution in both the
    # variational form and BC's, and we update it for each stage in
    # the loop over stages in the advance method.  The Constants a
    # and abar are used similarly in the variational form
    MC = MeshConstant(V.mesh(), backend=backend)
    if kgac is None:
        k0 = backend_cls.Function(V)
        g1 = backend_cls.Function(V)
        g2 = backend_cls.Function(V)
        a = MC.Constant(1.0)
        abar = MC.Constant(1.0)
        c = MC.Constant(1.0)
    else:
        k0, g1, g2, a, abar, c = kgac
    k = k0 if u0 == u else u

    repl = {t: t + c * dt,
            u: g1 + k * (abar * dt**2),
            Dt(u): g2 + k * (a * dt),
            Dt(u, 2): k}
    stage_F = replace(F, repl)

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
                gdat = bcarg_stage - backend_cls.bc2space(bc, u0) - c*dt*backend_cls.bc2space(bc, ut0)
                gdat -= sum(backend_cls.bc2space(bc, ks[i]) * (abar_vals[i] * dt**2) for i in range(num_stages))
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
                gdat = bcprime_stage - backend_cls.bc2space(bc, ut0)
                gdat -= sum(backend_cls.bc2space(bc, ks[i]) * (abar_vals[i] * dt) for i in range(num_stages))
                gdat /= d_val * dt
                bcnew.append(bc.reconstruct(g=gdat))

    return stage_F, (k0, g1, g2, a, abar, c), bcnew, (abar_vals, d_val)


class DIRKNystromTimeStepper:
    """Front-end class for advancing a second-order time-dependent PDE via a diagonally-implicit
    Runge-Kutta-Nystrom method formulated in terms of stage derivatives."""

    def __init__(self, F, tableau, t, dt, u0, ut0, bcs=None, J=None, Jp=None,
                 solver_parameters=None,
                 appctx=None, nullspace=None,
                 transpose_nullspace=None, near_nullspace=None,
                 bc_type=None,
                 backend: str = "firedrake",
                 **kwargs):
        backend_cls = get_backend(backend)
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

        self.AA = vecconst(tableau.A, backend=backend)
        self.AAbar = vecconst(tableau.Abar, backend=backend)
        self.BB = vecconst(tableau.b, backend=backend)
        self.BBbar = vecconst(tableau.bbar, backend=backend)
        self.CC = vecconst(tableau.c, backend=backend)

        if bc_type == "DAE":
            if tableau.is_explicit:
                raise NotImplementedError("Cannot have DAE BCs with Explicit Nystrom methods")
            self.AABbar = vecconst(tableau.Abar, backend=backend)
            self.CCone = vecconst(tableau.c, backend=backend)
        elif bc_type == "dDAE":
            if tableau.is_explicit:
                AABbar = numpy.vstack((tableau.A, tableau.b))
                self.AABbar = vecconst(AABbar[1:], backend=backend)
                CCone = numpy.append(tableau.c[1:], 1.0)
                self.CCone = vecconst(CCone, backend=backend)
            else:
                self.AABbar = vecconst(tableau.A, backend=backend)
                self.CCone = vecconst(tableau.c, backend=backend)
        else:
            raise NotImplementedError(f"No implementation for bc_type {bc_type} for DIRK-Nystrom or Explicit-Nystrom methods")

        V = backend_cls.get_function_space(u0)
        self.V = V
        self.u0 = u0
        self.ut0 = ut0
        self.t = t
        self.dt = dt
        self.num_fields = len(V)
        self.orig_bcs = bcs
        self.ks = [backend_cls.Function(V) for _ in range(num_stages)]

        stage_F, self.kgac, bcnew, (abar_vals, d_val) = getFormDIRKNystrom(
            F, self.ks, tableau, t, dt, u0, ut0, bcs=bcs, backend=backend)

        k = self.kgac[0]
        self.bcnew = bcnew

        stage_J = self.get_bilinear_form(J, u0, self.ks)
        stage_Jp = self.get_bilinear_form(Jp, u0, self.ks)

        appctx_irksome = {"stepper": self}
        if appctx is None:
            appctx = appctx_irksome
        else:
            appctx = {**appctx, **appctx_irksome}
        self.appctx = appctx

        self.problem = backend_cls.create_variational_problem(
            stage_F, k, bcs=bcnew, J=stage_J, Jp=stage_Jp,
            form_compiler_parameters=kwargs.pop("form_compiler_parameters", None),
            is_linear=kwargs.pop("is_linear", False),
            restrict=kwargs.pop("restrict", False),
        )
        constant_jacobian = kwargs.pop("constant_jacobian", False)
        if constant_jacobian:
            raise ValueError("Cannot set constant_jacobian=True on a DIRK")

        self.solver = backend_cls.create_variational_solver(
            self.problem, appctx=appctx,
            nullspace=nullspace,
            transpose_nullspace=transpose_nullspace,
            near_nullspace=near_nullspace,
            solver_parameters=solver_parameters,
            **kwargs,
        )

        self.bc_constants = abar_vals, d_val

    def get_bilinear_form(self, form, u0, stages, tableau=None):
        if form is None:
            return form
        Fbig, *_ = self.get_form_and_bcs(u0, stages, F=form, bcs=(), tableau=tableau)
        is_bilinear = len(Fbig.arguments()) == 2
        return lhs(Fbig) if is_bilinear else self._backend.derivative(Fbig, stages)

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

    def get_form_and_bcs(self, stages, F=None, bcs=None, tableau=None):
        if bcs is None:
            bcs = self.orig_bcs
        return getFormDIRKNystrom(F or self.F,
                                  stages,
                                  tableau or self.tableau,
                                  self.t, self.dt,
                                  self.u0, self.ut0,
                                  bcs=bcs, kgac=self.kgac)

    def invalidate_jacobian(self):
        """
        Forces the matrix to be reassembled next time it is required.
        """
        self._backend.invalidate_jacobian(self.solver)


class ExplicitNystromTimeStepper(DIRKNystromTimeStepper):
    """Front-end class for advancing a second-order time-dependent PDE via an explicit
    Runge-Kutta-Nystrom method formulated in terms of stage derivatives."""

    def __init__(self, F, tableau, t, dt, u0, ut0, bcs=None,
                 solver_parameters=None,
                 appctx=None, nullspace=None,
                 transpose_nullspace=None, near_nullspace=None,
                 bc_type=None,
                 backend: str = "firedrake",
                 **kwargs):
        if not isinstance(tableau, NystromTableau):
            tableau = butcher_to_nystrom(tableau)
        assert tableau.is_explicit
        if bc_type is None:
            bc_type = "dDAE"
        # we just have one mass matrix we're reusing for each time step and
        # each stage, so we can nudge this along
        solver_parameters = {} if solver_parameters is None else dict(solver_parameters)
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
            backend=backend,
            **kwargs)
