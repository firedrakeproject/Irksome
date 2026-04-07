"""Mass conservation test for nonlinear time derivatives.

Solves a Richards-like diffusion on a unit square with DQ1 elements and
an exponential soil model (no G-ADOPT dependency).  A constant flux is
applied on the top boundary; all other boundaries are no-flux.

The conserved quantity is moisture content theta(h), a nonlinear function
of pressure head h.  The test checks that the change in total moisture
matches the boundary flux to machine precision.

With the chain-rule expansion Dt(theta(h)) -> C(h)*Dt(h), the mass
balance error accumulates systematically over time.  With a conservative
discretisation that evaluates theta at the stage values directly, the
error stays near machine precision.

The conservative form requires a stiffly accurate method (where
u_new = U_s, the last stage value) so that the mass change
theta(u_new) - theta(u_old) is directly represented in the stage
equations.
"""
import pytest
from firedrake import (
    CellVolume, FacetArea, FacetNormal, Function,
    FunctionSpace, TestFunction, UnitSquareMesh,
    assemble, avg, ds, dS, dx, exp, grad, inner, jump,
)
from irksome import BackwardEuler, Dt, MeshConstant, RadauIIA, TimeStepper


# -- Exponential soil model (inlined from G-ADOPT) -----------------------
# theta(h) = theta_r + (theta_s - theta_r) * exp(alpha * h)   for h <= 0
# K(h)     = Ks * exp(alpha * h)                               for h <= 0

THETA_R = 0.15
THETA_S = 0.45
ALPHA = 0.328
KS = 1e-5


def theta(h):
    return THETA_R + (THETA_S - THETA_R) * exp(ALPHA * h)


def conductivity(h):
    return KS * exp(ALPHA * h)


# -- Problem parameters --------------------------------------------------

GRID = 10
DT_VAL = 100.0
STEPS = 50
FLUX = 1e-6
SIGMA = 10.0

SOLVER_PARAMS = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_atol": 1e-14,
}


def run_richards(butcher_tableau):
    """Run Richards problem and return cumulative mass balance error."""
    mesh = UnitSquareMesh(GRID, GRID, quadrilateral=True)
    V = FunctionSpace(mesh, "DQ", 1)
    h = Function(V, name="h").assign(-1.0)
    v = TestFunction(V)
    n = FacetNormal(mesh)

    MC = MeshConstant(mesh)
    t = MC.Constant(0.0)
    dt = MC.Constant(DT_VAL)

    # Mass term: Dt(theta(h)) -- should be conservative
    F = inner(v, Dt(theta(h))) * dx

    # SIPG diffusion
    K = conductivity(h)
    sigma_int = SIGMA * avg(FacetArea(mesh) / CellVolume(mesh))
    F += inner(grad(v), K * grad(h)) * dx
    F += sigma_int * inner(jump(v, n), avg(K) * jump(h, n)) * dS
    F -= inner(avg(K * grad(v)), jump(h, n)) * dS
    F -= inner(jump(v, n), avg(K * grad(h))) * dS

    # Flux BC on top (ds(4) is top boundary on UnitSquareMesh)
    F -= FLUX * v * ds(4)

    stepper = TimeStepper(F, butcher_tableau, t, dt, h,
                          solver_parameters=SOLVER_PARAMS,
                          stage_type="value")

    mass_expr = interpolate(theta(h), V) * dx
    cum_error = 0.0

    curr_mass = assemble(mass_expr)
    for step in range(STEPS):
        prev_mass = curr_mass

        stepper.advance()
        t.assign(float(t) + float(dt))

        curr_mass = assemble(mass_expr)

        expected_change = DT_VAL * FLUX
        cum_error += abs(abs(curr_mass - prev_mass) - expected_change)

    return cum_error


@pytest.mark.parametrize("tableau", [BackwardEuler(), RadauIIA(2)],
                         ids=["BackwardEuler", "RadauIIA2"])
def test_mass_conservation_stage_value(tableau):
    """Dt(theta(h)) with stage_type='value' and a stiffly accurate
    method should conserve mass."""
    err = run_richards(tableau)
    assert err < 1e-10, (
        f"Stage value stepper mass error should be near machine precision, "
        f"got {err:.2e}"
    )
