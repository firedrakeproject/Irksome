import pytest
from firedrake import *
from irksome import Dt, GaussLegendre, SSPButcherTableau, MeshConstant, TimeStepper


@pytest.fixture
def heat_problem():
    """Simple heat equation setup."""
    msh = UnitSquareMesh(4, 4)
    MC = MeshConstant(msh)
    V = FunctionSpace(msh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx
    return F, MC.Constant(0), MC.Constant(0.1), u


@pytest.mark.parametrize("stage_type,tableau", [
    ("deriv", GaussLegendre(1)),
    ("value", GaussLegendre(1)),
    ("dirk", GaussLegendre(1)),
    ("explicit", SSPButcherTableau(2, 2)),
])
def test_base_kwargs(heat_problem, stage_type, tableau):
    """Test that valid_base_kwargs are passed through for all stage types."""
    F, t, dt, u = heat_problem
    stepper = TimeStepper(
        F, tableau, t, dt, u,
        stage_type=stage_type,
        options_prefix="test_prefix_",
        form_compiler_parameters={"quadrature_degree": 4},
    )
    assert stepper.solver.snes.getOptionsPrefix() == "test_prefix_"


@pytest.mark.parametrize("stage_type,tableau", [
    ("deriv", GaussLegendre(1)),
    ("value", GaussLegendre(1)),
    ("dirk", GaussLegendre(1)),
])
@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_derivative_jacobian(heat_problem, stage_type, tableau, nonlinear):
    """A Jacobian built by derivative() must step the same as the default one."""
    F, t, dt, u = heat_problem
    V = u.function_space()
    x, y = SpatialCoordinate(V.mesh())
    ic = sin(pi * x) * sin(pi * y)
    if nonlinear:
        F += inner(u ** 3, TestFunction(V)) * dx

    def step(**kwargs):
        t.assign(0.0)
        u.interpolate(ic)
        stepper = TimeStepper(F, tableau, t, dt, u,
                              stage_type=stage_type, **kwargs)
        stepper.advance()
        return u.copy(deepcopy=True)

    expect = step()
    got = step(J=derivative(F, u))
    assert errornorm(expect, got) < 1e-8


def test_base_kwargs_adaptive(heat_problem):
    """Test that valid_base_kwargs are passed through for adaptive stepper."""
    F, t, dt, u = heat_problem
    stepper = TimeStepper(
        F, GaussLegendre(2), t, dt, u,
        adaptive_parameters={"tol": 1e-2},
        options_prefix="test_adaptive_",
        form_compiler_parameters={"quadrature_degree": 4},
    )
    assert stepper.solver.snes.getOptionsPrefix() == "test_adaptive_"
