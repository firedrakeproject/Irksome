from firedrake import *
from firedrake.adjoint import *
from irksome import Alexander, Dt, RadauIIA, GaussLegendre, TimeStepper, WSODIRK
from irksome.backend import get_backend
import pytest


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    if annotate_tape():
        pause_annotation()


@pytest.mark.parametrize("alpha", [0, 1])
def test_adjoint_mixed_stage_operations(tuple_subfunctions, alpha):
    backend = get_backend("firedrake")
    mesh = UnitSquareMesh(1, 1)
    V = VectorFunctionSpace(mesh, "CG", 1) * FunctionSpace(mesh, "DG", 0)
    m = Function(V).assign(2)
    control = Control(m)
    stages = backend.get_stages(V, 3)
    backend.assign_stages(stages, m, [1, -2, 3])
    u = Function(V).assign(0.5 * m)
    backend.assign_stage_sum(u, stages, [2, 0, -1], alpha=alpha)
    J = assemble(inner(u, u) * dx)
    rf = ReducedFunctional(J, control)
    direction = Function(V).assign(0.5)
    assert taylor_test(rf, m, direction) > 1.95


@pytest.mark.parametrize("nt", (pytest.param(n, id=f"nt={n}") for n in (1, 4)))
@pytest.mark.parametrize("stage_type,bt",
                         [(stage, ibt(k)) for ibt in (RadauIIA, GaussLegendre)
                          for k in (1, 3) for stage in ("value", "deriv")]
                         + [("dirk", blah) for blah in (Alexander(), WSODIRK(4, 3, 3))])
def test_adjoint_diffusivity(nt, stage_type, bt):
    msh = UnitIntervalMesh(8)
    x, = SpatialCoordinate(msh)
    V = FunctionSpace(msh, "CG", 1)

    R = FunctionSpace(msh, "R", 0)
    kappa = Function(R).assign(2.0)

    v = TestFunction(V)
    u = Function(V)
    u.interpolate(sin(pi*x))

    dt = Constant(0.1)
    t = Constant(0.)

    bcs = DirichletBC(V, 0., "on_boundary")
    F = inner(Dt(u), v) * dx + kappa * inner(grad(u), grad(v)) * dx
    stepper = TimeStepper(F, bt, t, dt, u, bcs=bcs, stage_type=stage_type)

    continue_annotation()
    with set_working_tape() as tape:
        for _ in range(nt):
            stepper.advance()
        J = assemble(inner(u, u) * dx)
        rf = ReducedFunctional(J, Control(kappa), tape=tape)
    pause_annotation()

    taylor = taylor_to_dict(rf, kappa, Constant(-1.0))
    assert min(taylor['R0']['Rate']) > 0.95, taylor['R0']
    assert min(taylor['R1']['Rate']) > 1.95, taylor['R1']
    assert min(taylor['R2']['Rate']) > 2.95, taylor['R2']
