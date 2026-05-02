import pytest
import firedrake
from firedrake import Constant, inner, grad, dx, conditional
import irksome
from irksome import Dt
from irksome.ufl.deriv import lag


@pytest.mark.xfail(strict=True, reason="lag not yet honored by replace")
def test_stefan_implicit():
    """Test lagging the conductivity on the Stefan problem"""
    nx = 32
    mesh = firedrake.UnitIntervalMesh(nx)
    V = firedrake.FunctionSpace(mesh, "CG", 1)
    x, = firedrake.SpatialCoordinate(mesh)

    u = firedrake.Function(V)
    u.interpolate(1 - 2 * x)

    k_solid = Constant(2.0)
    k_liquid = Constant(1.0)
    k = lag(conditional(u < 0, k_solid, k_liquid))

    v = firedrake.TestFunction(V)
    F = (Dt(u) * v + k * inner(grad(u), grad(v))) * dx

    T_1 = Constant(1.0)
    T_2 = Constant(-1.0)
    bcs = [firedrake.DirichletBC(V, T_1, 1), firedrake.DirichletBC(V, T_2, 2)]

    t = Constant(0.0)
    dt = Constant(0.1)
    solver_params = {"snes_type": "newtonls", "snes_converged_reason": None}
    params = {"bcs": bcs, "solver_parameters": solver_params}
    method = irksome.BackwardEuler()
    stepper = irksome.TimeStepper(F, method, t, dt, u, **params)

    final_time = 10.0
    num_steps = int(final_time / float(dt))
    for step in range(num_steps):
        stepper.advance()
