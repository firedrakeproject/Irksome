"""Check mass conservation for Dt(theta(h)) with nonlinear theta.

Richards-like diffusion on a unit square, CG1, exponential soil
model, pure Neumann BCs.  The total moisture should change by
exactly the boundary flux each step.  Stiffly accurate methods
give u_new = U_s, so the conservative finite difference appears
directly in the stage equations.
"""
import pytest
from firedrake import (
    Constant, Function, FunctionSpace, TestFunction,
    UnitSquareMesh, assemble, ds, dx, exp, grad, inner,
)
from irksome import BackwardEuler, DiscontinuousGalerkinScheme, Dt, RadauIIA, TimeStepper


def run_richards(scheme, **kwargs):
    """Run the problem and return cumulative mass balance error."""
    # Exponential soil model
    theta_r = 0.15
    theta_s = 0.45
    alpha = 0.328
    Ks = 1e-5
    theta = lambda h: theta_r + (theta_s - theta_r) * exp(alpha * h)
    conductivity = lambda h: Ks * exp(alpha * h)

    N = 10
    dt_val = 100
    nstep = 50
    flux = 1e-6

    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "CG", 1)
    h = Function(V, name="h").assign(-1.0)
    v = TestFunction(V)

    t = Constant(0.0)
    dt = Constant(dt_val)

    F = inner(Dt(theta(h)), v) * dx
    F += inner(conductivity(h) * grad(h), grad(v)) * dx
    F -= inner(flux, v) * ds(4)

    stepper = TimeStepper(F, scheme, t, dt, h,
                          solver_parameters={
                              "snes_atol": 1e-14,
                          },
                          **kwargs)

    mass_form = theta(h) * dx
    curr_mass = assemble(mass_form)
    cum_error = 0.0

    for step in range(nstep):
        prev_mass = curr_mass
        stepper.advance()
        t.assign(float(t) + float(dt))
        curr_mass = assemble(mass_form)
        cum_error += abs(abs(curr_mass - prev_mass) - float(dt) * flux)

    return cum_error


@pytest.mark.parametrize("scheme", [BackwardEuler(), RadauIIA(2)],
                         ids=["BackwardEuler", "RadauIIA2"])
def test_mass_conservation_stage_value(scheme):
    """Test mass conservation with Dt(theta(h))"""
    err = run_richards(scheme, stage_type="value")
    assert err < 1e-10, (
        f"mass error should be near machine precision, got {err:.2e}"
    )


@pytest.mark.parametrize("order", [0, 1])
def test_mass_conservation_dg(order):
    """Test mass conservation with Dt(theta(h))"""
    scheme = DiscontinuousGalerkinScheme(order)
    err = run_richards(scheme)
    assert err < 1e-10, (
        f"mass error should be near machine precision, got {err:.2e}"
    )
