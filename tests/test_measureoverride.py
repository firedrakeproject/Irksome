import pytest

from firedrake import *
from irksome import Dt, TimeStepper, ContinuousPetrovGalerkinScheme, dx_override

@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("scheme", ["gauss", "cpg"])
def test_nls(order, scheme):
    # Domain and space
    mesh = PeriodicUnitIntervalMesh(10)
    x, = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    Z = V * V

    # State and test functions
    psi = Function(Z)
    a, b = split(psi)
    c, d = TestFunctions(Z)

    # Initial condition: cosine
    psi.project(as_vector([cos(x), 0]))

    # Time parameters
    t = Constant(0.0)
    dt = Constant(0.1)

    # Residual
    dx_highorder = dx if scheme == "gauss" else dx_override(time_degree_override=4*order-1)
    amp_sq = a**2 + b**2
    F = (
        inner(Dt(b), c) * dx
        + 0.5 * inner(grad(a), grad(c)) * dx
        - inner(amp_sq * a, c) * dx_highorder
        - inner(Dt(a), d) * dx
        + 0.5 * inner(grad(b), grad(d)) * dx
        - inner(amp_sq * b, d) * dx_highorder
    )

    # Energy
    E = 0.5 * (inner(grad(a), grad(a)) + inner(grad(b), grad(b)) - amp_sq**2) * dx

    # Time stepper with cPG(k); default time quadrature is 2k-1
    scheme_ = ContinuousPetrovGalerkinScheme(order=order, quadrature_degree=2*order-1)
    stepper = TimeStepper(F, scheme_, t, dt, psi)

    # Record initial energy
    E0 = float(assemble(E))

    # Advance once
    stepper.advance()

    # Final energy and drift
    E1 = float(assemble(E))
    drift = abs(E1 - E0)
    if scheme == "gauss": assert drift > 1e-10
    else: assert drift < 1e-10
