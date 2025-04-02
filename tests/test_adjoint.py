from firedrake import *
from firedrake.adjoint import *
from irksome import Dt, RadauIIA, GaussLegendre, TimeStepper
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


@pytest.mark.parametrize("nt", (pytest.param(n, id=f"nt={n}") for n in (1, 4)))
@pytest.mark.parametrize("stages", (pytest.param(n, id=f"stages={n}") for n in (1, 3)))
@pytest.mark.parametrize("Scheme", (RadauIIA, GaussLegendre))
def test_adjoint_diffusivity(nt, stages, Scheme):
    msh = UnitIntervalMesh(8)
    x, = SpatialCoordinate(msh)
    V = FunctionSpace(msh, "CG", 1)

    R = FunctionSpace(msh, "R", 0)
    kappa = Function(R).assign(2.0)

    v = TestFunction(V)
    u = Function(V)
    u.interpolate(sin(pi*x))

    dt = Constant(0.1)
    t = Constant(0)
    bt = Scheme(stages)

    bcs = DirichletBC(V, 0, "on_boundary")
    F = inner(Dt(u), v) * dx + kappa * inner(grad(u), grad(v)) * dx
    stepper = TimeStepper(F, bt, t, dt, u, bcs=bcs)

    continue_annotation()
    with set_working_tape() as tape:
        for _ in range(nt):
            stepper.advance()
        J = assemble(inner(u, u) * dx)
        rf = ReducedFunctional(J, Control(kappa), tape=tape)
    pause_annotation()
    assert taylor_test(rf, kappa, Constant(-1.0)) > 1.9
