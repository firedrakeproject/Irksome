import pytest
from petsc4py import PETSc
import firedrake
from firedrake import (
    Constant, DirichletBC, Function, FunctionSpace, MixedVectorSpaceBasis,
    SpatialCoordinate, TestFunction, TestFunctions, UnitIntervalMesh,
    UnitSquareMesh, VectorFunctionSpace, VectorSpaceBasis, as_vector, assemble,
    cos, div, dot, ds, dx, errornorm, exp, grad, inner, norm, replace, solve, split
)
from irksome import (
    BackwardEuler, Dt, IRKAuxiliaryOperatorSNES, RadauIIA, RanaLDScheme,
    TimeStepper, lag,
)

# Stefan problem: a two-phase heat equation with a discontinuous conductivity
#
#   k(T) = k_solid  if T <= T_m  else  k_liquid.
#
# The fully implicit residual is not differentiable in T, so Newton's method
# fails. Lagging the conductivity to the start of the timestep gives a residual
# with a well-defined Jacobian. Here we use the lagged residual as a nonlinear
# preconditioner so that the outer solver converges to the fully implicit
# problem.

T_m = Constant(0.0)
k_solid = Constant(2.0)
k_liquid = Constant(1.0)
h = Constant(0.01)            # Robin penalty length scale
dT = Constant(0.25)           # amplitude of the boundary oscillation
T_scale = Constant(1e-3)      # scale for transition between solid and liquid


def stefan_conductivity(T):
    r = exp(-T / T_scale)
    s = 1 / (1 + r)
    return k_solid * (1 - s) + s * k_liquid


def stefan_form(T, q, t, k):
    """Robin residual with the oscillating boundary temperatures."""
    T_1 = Constant(-1.0)
    T_2 = Constant(1.0) + dT * cos(t)
    F_cells = (Dt(T) * q + k * inner(grad(T), grad(q))) * dx
    F_boundaries = k / h * (T - T_1) * q * ds(1) + k / h * (T - T_2) * q * ds(2)
    return F_cells + F_boundaries


class StefanLagAuxSNES(IRKAuxiliaryOperatorSNES):
    """Precondition the fully implicit Stefan problem with the residual whose
    conductivity is lagged to the start of the timestep."""
    def getNewForm(self, snes, T, q):
        t = self.get_appctx(snes)["stepper"].t
        k = lag(stefan_conductivity(T))
        return stefan_form(T, q, t, k), None


def stefan_setup():
    nx = 64
    mesh = UnitIntervalMesh(nx)
    Q = FunctionSpace(mesh, "CG", 1)
    x, = SpatialCoordinate(mesh)

    T = Function(Q)
    T.interpolate((1 - x) * Constant(-1.0) + x * Constant(1.0))

    t = Constant(0.0)
    dt = Constant(0.1)
    q = TestFunction(Q)
    k = stefan_conductivity(T)   # NOTE: this is the fully implicit form
    F = stefan_form(T, q, t, k)
    return F, t, dt, T


def test_npc_stefan_newton_fails():
    """Newton's method fails on the fully implicit Stefan problem because it
    isn't differentiable."""
    F, t, dt, T = stefan_setup()
    method = BackwardEuler()
    params = {
        "solver_parameters": {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_monitor": None,
        },
    }
    stepper = TimeStepper(F, method, t, dt, T, **params)
    with pytest.raises(firedrake.ConvergenceError):
        final_time = 20.0
        num_steps = int(final_time / float(dt))
        for step in range(num_steps):
            stepper.advance()
            t.assign(float(t) + float(dt))


@pytest.mark.skipif(
    not hasattr(firedrake, "AuxiliaryOperatorSNES"),
    reason="Firedrake does not provide AuxiliaryOperatorSNES",
)
def test_npc_stefan():
    """The lagged residual, used as a nonlinear preconditioner, drives the
    fully implicit Stefan problem to convergence where Newton fails."""
    F, t, dt, T = stefan_setup()
    params = {
        "solver_parameters": {
            "snes_type": "ngmres",
            "snes_monitor": None,
            "snes_max_it": 100,
            "npc": {
                "snes_type": "python",
                "snes_python_type": "test_npc.StefanLagAuxSNES",
                "aux": {
                    "snes_type": "fieldsplit",
                    "snes_fieldsplit_0_fields": "0",
                    "snes_max_it": 1,
                    "snes_convergence_test": "skip",
                    "fieldsplit_0": {
                        "snes_type": "ksponly",
                        "ksp_type": "cg",
                        "pc_type": "lu",
                    },
                },
            },
        },
    }
    method = BackwardEuler()
    stepper = TimeStepper(F, method, t, dt, T, **params)

    reasons = [
        PETSc.SNES.ConvergedReason.CONVERGED_FNORM_ABS,
        PETSc.SNES.ConvergedReason.CONVERGED_FNORM_RELATIVE,
    ]

    final_time = 20.0
    num_steps = int(final_time / float(dt))
    for step in range(num_steps):
        stepper.advance()
        t.assign(t + dt)
        assert stepper.solver.snes.getConvergedReason() in reasons


@pytest.fixture(params=[False, True], ids=["scalar", "mixed"])
def stage_split_problem(request):
    mesh = UnitIntervalMesh(8)
    V = FunctionSpace(mesh, "CG", 1)
    Z = V * V if request.param else V
    u = Function(Z)
    v = TestFunction(Z)
    x, = SpatialCoordinate(mesh)
    t = Constant(0.0)
    dt = Constant(0.1)
    for i, component in enumerate(u.subfunctions):
        component.interpolate((i + 1) * (1 + x))
    F = (inner(Dt(u), v) + inner(grad(u), grad(v)) + inner(inner(u, u) * u, v)) * dx
    Fp = (inner(Dt(u), v) + inner(grad(u), grad(v)) + inner(u, v)) * dx
    bcs = [DirichletBC(Z.sub(i), (i + 1) * (1 + x + t), "on_boundary") for i in range(len(Z))]
    return F, Fp, t, dt, u, bcs


@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("split_type", ["additive", "multiplicative"])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_stage_split_snes(stage_split_problem, num_stages, split_type, auxiliary):
    F, Fp, t, dt, u, bcs = stage_split_problem
    initial = u.copy(deepcopy=True)
    scheme = RadauIIA(num_stages)
    direct_parameters = {
        "snes_rtol": 1e-11,
        "snes_atol": 1e-11,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    reference_stepper = TimeStepper(F, scheme, t, dt, u, bcs=bcs, solver_parameters=direct_parameters)
    for _ in range(2):
        reference_stepper.advance()
        t.assign(t + dt)
    reference = u.copy(deepcopy=True)

    u.assign(initial)
    t.assign(0)
    split_parameters = {
        "snes_type": "fieldsplit",
        "snes_fieldsplit_type": split_type,
        "snes_max_it": 1,
        "snes_convergence_test": "skip",
        "fieldsplit": direct_parameters,
    }
    nf = len(u.function_space())
    split_parameters.update({
        f"snes_fieldsplit_{i}_fields": ",".join(map(str, range(i * nf, (i + 1) * nf)))
        for i in range(num_stages)
    })
    npc_parameters = {
        "snes_type": "python",
        "snes_python_type": "irksome.IRKAuxiliaryOperatorSNES",
        "snes_max_it": 1,
        "snes_convergence_test": "skip",
        "aux": split_parameters,
    } if auxiliary else split_parameters
    solver_parameters = {
        "snes_type": "ngmres",
        "snes_rtol": 1e-11,
        "snes_atol": 1e-11,
        "snes_max_it": 100,
        "npc": npc_parameters,
    }
    appctx = {"Fp": Fp, "Fp_scheme": RanaLDScheme(scheme)} if auxiliary else None
    stepper = TimeStepper(F, scheme, t, dt, u, bcs=bcs, solver_parameters=solver_parameters, appctx=appctx)
    stepper.advance()
    npc = stepper.solver.snes.getNPC()
    stage_split = npc.getPythonContext().solver.snes if auxiliary else npc
    children = stage_split.getFieldSplitSubSNES()
    assert [stage_split.getFieldSplitFields(i) for i in range(num_stages)] == [
        tuple(range(i * nf, (i + 1) * nf)) for i in range(num_stages)]
    assert [child.getOptionsPrefix() for child in children] == [
        f"{stage_split.getOptionsPrefix()}fieldsplit_{i}_" for i in range(num_stages)]
    t.assign(t + dt)
    stepper.advance()
    assert stage_split.getFieldSplitSubSNES() == children
    assert errornorm(reference, u) < 1e-9


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("split_type", ["additive", "multiplicative"])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_stage_split_snes_parallel(stage_split_problem, split_type, auxiliary):
    test_stage_split_snes(stage_split_problem, 2, split_type, auxiliary)


def test_stage_split_frozen_auxiliary_sweep():
    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V).assign(1)
    v = TestFunction(V)
    t = Constant(0)
    dt = Constant(0.5)
    F = (Dt(u) * v + u**3 * v) * dx
    scheme = RadauIIA(2)
    auxiliary_scheme = RanaLDScheme(scheme)
    direct_parameters = {
        "snes_atol": 1e-12,
        "snes_rtol": 1e-12,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    parameters = {
        "snes_type": "python",
        "snes_python_type": "irksome.IRKAuxiliaryOperatorSNES",
        "snes_max_it": 1,
        "snes_convergence_test": "skip",
        "aux": {
            "snes_type": "fieldsplit",
            "snes_fieldsplit_type": "multiplicative",
            "snes_fieldsplit_0_fields": "0",
            "snes_fieldsplit_1_fields": "1",
            "snes_max_it": 1,
            "snes_convergence_test": "skip",
            "fieldsplit": direct_parameters,
        },
    }
    stepper = TimeStepper(F, scheme, t, dt, u, stage_type="value",
                          solver_parameters=parameters,
                          appctx={"Fp_scheme": auxiliary_scheme})
    reference = stepper.stages.copy(deepcopy=True)
    G, _ = stepper.get_form_and_bcs(reference, tableau=auxiliary_scheme)
    forcing = assemble(replace(G, {reference: stepper.stages}) - stepper.problem.F)
    solve(G - forcing == 0, reference, solver_parameters=direct_parameters)

    # A forward sweep solves the triangular auxiliary equation with the
    # forcing frozen at the original stage iterate.
    stepper.solver.solve()
    assert errornorm(reference, stepper.stages) < 1e-11


def test_stage_split_snes_navier_stokes():
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    Z = V * Q
    up = Function(Z)
    u, p = split(up)
    v, q = TestFunctions(Z)
    x, y = SpatialCoordinate(mesh)
    t = Constant(0.0)
    dt = Constant(0.05)
    Fp = (inner(Dt(u), v) + 0.1 * inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q)) * dx
    F = Fp + inner(dot(grad(u), u), v) * dx
    bcs = [DirichletBC(Z.sub(0), 0, (1, 2, 3)),
           DirichletBC(Z.sub(0), as_vector((16 * x**2 * (1 - x)**2, 0)), 4)]
    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=mesh.comm)])
    scheme = RadauIIA(2)
    direct_parameters = {
        "snes_rtol": 1e-10,
        "snes_atol": 1e-10,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    reference_stepper = TimeStepper(F, scheme, t, dt, up, bcs=bcs, nullspace=nullspace, solver_parameters=direct_parameters)
    for _ in range(3):
        reference_stepper.advance()
        t.assign(t + dt)
    reference = up.copy(deepcopy=True)

    up.assign(0)
    t.assign(0)
    split_parameters = {
        "snes_type": "fieldsplit",
        "snes_fieldsplit_type": "multiplicative",
        "snes_max_it": 1,
        "snes_convergence_test": "skip",
        "snes_fieldsplit_0_fields": "0,1",
        "snes_fieldsplit_1_fields": "2,3",
        "fieldsplit": direct_parameters,
    }
    solver_parameters = {
        "snes_type": "ngmres",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-10,
        "snes_max_it": 100,
        "npc": {
            "snes_type": "python",
            "snes_python_type": "irksome.IRKAuxiliaryOperatorSNES",
            "snes_max_it": 1,
            "snes_convergence_test": "skip",
            "aux": split_parameters,
        },
    }
    appctx = {"Fp": Fp, "Fp_scheme": RanaLDScheme(scheme)}
    stepper = TimeStepper(F, scheme, t, dt, up, bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters, appctx=appctx)
    for _ in range(3):
        stepper.advance()
        t.assign(t + dt)
    ur, pr = reference.subfunctions
    ua, pa = up.subfunctions
    pressure_error = pr - pa - assemble((pr - pa) * dx)
    assert errornorm(ur, ua) < 1e-7
    assert norm(pressure_error) < 1e-7


@pytest.mark.parallel(nprocs=2)
def test_stage_split_snes_navier_stokes_parallel():
    test_stage_split_snes_navier_stokes()
