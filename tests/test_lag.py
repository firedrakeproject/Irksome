import pytest
import firedrake
from firedrake import Constant, inner, grad, dx, conditional, assemble, replace
import irksome
from irksome import Dt, lag


def test_stefan_implicit():
    """Test lagging the conductivity on the Stefan problem"""
    nx = 32
    mesh = firedrake.UnitIntervalMesh(nx)
    V = firedrake.FunctionSpace(mesh, "CG", 1)
    x, = firedrake.SpatialCoordinate(mesh)

    u = firedrake.Function(V)
    u.interpolate(1 - 2 * x)
    u_0 = u.copy(deepcopy=True)

    T_1 = Constant(1.0)
    T_2 = Constant(-1.0)
    bcs = [firedrake.DirichletBC(V, T_1, 1), firedrake.DirichletBC(V, T_2, 2)]

    v = firedrake.TestFunction(V)
    t = Constant(0.0)
    dt = Constant(0.1)
    final_time = 10.0
    num_steps = int(final_time / float(dt))
    solver_params = {"snes_type": "newtonls", "snes_converged_reason": None}
    params = {"bcs": bcs, "solver_parameters": solver_params}
    method = irksome.BackwardEuler()

    k_solid = Constant(2.0)
    k_liquid = Constant(1.0)

    # Check that fully implicit form fails
    k = conditional(u < 0, k_solid, k_liquid)
    F = (Dt(u) * v + k * inner(grad(u), grad(v))) * dx
    stepper = irksome.TimeStepper(F, method, t, dt, u, **params)
    with pytest.raises(firedrake.ConvergenceError):
        for step in range(num_steps):
            stepper.advance()

    # Check that the lagged form works
    u.assign(u_0)
    F = (Dt(u) * v + lag(k) * inner(grad(u), grad(v))) * dx
    stepper = irksome.TimeStepper(F, method, t, dt, u, **params)

    for step in range(num_steps):
        stepper.advance()

    energy = 0.5 * u**2 * dx
    initial_energy = assemble(replace(energy, {u: u_0}))
    final_energy = assemble(energy)
    print(f"Initial energy: {initial_energy}")
    print(f"Final:          {final_energy}")
    assert final_energy <= initial_energy
