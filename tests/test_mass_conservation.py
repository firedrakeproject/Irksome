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
import numpy as np


def run_richards(scheme, **kwargs):
    """Run the problem and return mean mass balance error."""
    # Exponential soil model
    theta_r = Constant(0.15)
    theta_s = Constant(0.45)
    alpha = Constant(0.328)
    Ks = Constant(1e-5)

    theta = lambda h: theta_r + (theta_s - theta_r) * exp(alpha * h)
    conductivity = lambda h: Ks * exp(alpha * h)

    N = 10
    dt_val = 100
    nstep = 50

    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "CG", 1)
    h = Function(V, name="h").assign(-1.0)
    v = TestFunction(V)

    t = Constant(0.0)
    dt = Constant(dt_val)
    flux = Constant(1e-6)

    F = inner(Dt(theta(h)), v) * dx
    F += inner(conductivity(h) * grad(h), grad(v)) * dx
    F -= inner(flux, v) * ds(4)

    stepper = TimeStepper(F, scheme, t, dt, h,
                          solver_parameters={
                              "snes_rtol": 1e-10,
                              "snes_atol": 1e-14,
                          },
                          **kwargs)

    area = assemble(1*ds(4, domain=mesh))
    # Interpolate a Constant into V to compute mass with the same quadrature rule
    one = Function(V).interpolate(Constant(1))
    mass_form = inner(theta(h), one) * dx
    curr_mass = assemble(mass_form)
    total_error = 0.0

    for step in range(nstep):
        if step > 0 and step % 10 == 0:
            # Update the time step and the flux
            dt.assign(0.5*dt)
            flux.assign(4*flux)
        expected_mass = curr_mass + area * float(flux) * float(dt)
        stepper.advance()
        t.assign(t + dt)
        curr_mass = assemble(mass_form)
        total_error += abs(curr_mass - expected_mass)

    mean_error = total_error / float(t)
    return mean_error


@pytest.mark.parametrize("scheme", [BackwardEuler(), RadauIIA(2)],
                         ids=["BackwardEuler", "RadauIIA2"])
def test_mass_conservation_stage_value(scheme):
    """Test mass conservation with Dt(theta(h))"""
    err = run_richards(scheme, stage_type="value")
    assert err < 10*np.finfo(np.dtype(err)).eps, (
        f"mass error should be near machine precision, got {err:.2e}"
    )


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("deriv_type", ["strong", "weak"])
def test_mass_conservation_dg(order, deriv_type):
    """Test mass conservation with Dt(theta(h))"""
    scheme = DiscontinuousGalerkinScheme(order, deriv_type=deriv_type)
    err = run_richards(scheme)
    assert err < 200*np.finfo(np.dtype(err)).eps, (
        f"mass error should be near machine precision, got {err:.2e}"
    )
