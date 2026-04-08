"""Check mass conservation for Dt(theta(h)) with nonlinear theta.

Richards-like diffusion on a unit square, CG1, exponential soil
model, pure Neumann BCs.  The total moisture should change by
exactly the boundary flux each step.  Stiffly accurate methods
give u_new = U_s, so the conservative finite difference appears
directly in the stage equations.
"""
import pytest
from firedrake import (
    Function, FunctionSpace, TestFunction,
    UnitSquareMesh, assemble, ds, dx, exp, grad, inner,
)
from irksome import BackwardEuler, Dt, MeshConstant, RadauIIA, TimeStepper


# Exponential soil model
theta_r = 0.15
theta_s = 0.45
alpha = 0.328
Ks = 1e-5

N = 10
dt_val = 100.0
nstep = 50
flux = 1e-6


def theta(h):
    return theta_r + (theta_s - theta_r) * exp(alpha * h)


def conductivity(h):
    return Ks * exp(alpha * h)


def run_richards(butcher_tableau, stage_type):
    """Run the problem and return cumulative mass balance error."""
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "CG", 1)
    h = Function(V, name="h").assign(-1.0)
    v = TestFunction(V)

    MC = MeshConstant(mesh)
    t = MC.Constant(0.0)
    dt = MC.Constant(dt_val)

    F = inner(Dt(theta(h)), v) * dx
    F += inner(conductivity(h) * grad(h), grad(v)) * dx
    F -= flux * v * ds(4)

    stepper = TimeStepper(F, butcher_tableau, t, dt, h,
                          solver_parameters={
                              "mat_type": "aij",
                              "snes_type": "newtonls",
                              "ksp_type": "preonly",
                              "pc_type": "lu",
                              "pc_factor_mat_solver_type": "mumps",
                              "snes_atol": 1e-14,
                          },
                          stage_type=stage_type)

    mass_form = theta(h) * dx
    curr_mass = assemble(mass_form)
    cum_error = 0.0

    for step in range(nstep):
        prev_mass = curr_mass
        stepper.advance()
        t.assign(float(t) + float(dt))
        curr_mass = assemble(mass_form)
        cum_error += abs(abs(curr_mass - prev_mass) - dt_val * flux)

    return cum_error


@pytest.mark.parametrize("tableau", [BackwardEuler(), RadauIIA(2)],
                         ids=["BackwardEuler", "RadauIIA2"])
def test_mass_conservation_stage_value(tableau):
    """Stage value stepper with Dt(theta(h)) should conserve mass."""
    err = run_richards(tableau, "value")
    assert err < 1e-10, (
        f"mass error should be near machine precision, got {err:.2e}"
    )


def test_mass_conservation_fails_stage_derivative():
    """Stage derivative must apply the chain rule, so the linearisation
    error means mass is not conserved to machine precision."""
    err = run_richards(BackwardEuler(), "deriv")
    assert err > 1e-8, (
        f"stage derivative should NOT conserve mass, got {err:.2e}"
    )
