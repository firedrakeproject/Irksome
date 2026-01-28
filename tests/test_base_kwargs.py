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
