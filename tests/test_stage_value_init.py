from firedrake import *
from irksome import Dt, TimeStepper, BackwardEuler, RadauIIA
import pytest


@pytest.mark.parametrize("basis_type", ["Lagrange", "Bernstein"])
@pytest.mark.parametrize("degree", [1, 2])
def test_stage_value_init(basis_type, degree):
    nx = 16
    lx = 1.0
    mesh = IntervalMesh(nx, lx)

    Q = FunctionSpace(mesh, "DG", 0)
    x, = SpatialCoordinate(mesh)

    p = Function(Q)
    p_in = Constant(0.5)
    p.assign(p_in)

    q = TestFunction(Q)

    u_0 = Constant(1.0)
    du = Constant(1.0)
    Lx = Constant(lx)
    u = as_vector((u_0 + du * x / Lx,))

    p_max = Constant(2.0)
    s = p_max / p - 1
    F_cells = (Dt(p) * q - inner(p * u, grad(q)) - s * q) * dx
    n = FacetNormal(mesh)
    f_n = p * max_value(0, inner(u, n))
    F_facets = (f_n("+") - f_n("-")) * (q("+") - q("-")) * dS
    F_inflow = p_in * min_value(0, inner(u, n)) * q * ds
    F_outflow = p * max_value(0, inner(u, n)) * q * ds
    F = F_cells + F_facets + F_inflow + F_outflow

    method = BackwardEuler() if degree == 1 else RadauIIA(degree)
    t = Constant(0.0)
    timestep = 0.5 / nx
    dt = Constant(timestep)
    params = {
        "stage_type": "value",
        "basis_type": basis_type,
        "solver_parameters": {"snes_monitor": None},
    }
    stepper = TimeStepper(F, method, t, dt, p, **params)

    final_time = 2.0
    num_steps = int(final_time / timestep)
    for step in range(num_steps):
        stepper.advance()

    assert norm(p) > 0.0
