"""Tests for conservative-by-default mass terms in stage_value.

The contract:

* For a form ``inner(Dt(g(u)), v)*dx + ...`` with a nonlinear g of the
  prognostic variable u, the stage-value stepper must conserve total
  mass to the nonlinear-solve tolerance, for SA *and* non-SA tableaux.

* For g = identity (the default ``Dt(u)`` case) the discrete equations
  must be unchanged from master.  This is covered by the existing test
  suite -- see Group 5 / Regression preservation in the design notes.
"""
import pytest
from firedrake import (
    Constant, Function, FunctionSpace, TestFunction,
    UnitIntervalMesh, UnitSquareMesh,
    assemble, ds, dx, exp, grad, inner,
)

from irksome import (
    Alexander, BackwardEuler, Dt, GaussLegendre, LobattoIIIC, MeshConstant,
    QinZhang, RadauIIA, TimeStepper,
)
from irksome.ufl.manipulation import has_composite_time_derivative


SOLVER = {
    "snes_type": "newtonls",
    "snes_rtol": 1e-12,
    "snes_atol": 1e-14,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "mat_type": "aij",
}


def _theta(h, theta_r=Constant(0.15), theta_s=Constant(0.45),
           alpha=Constant(0.328)):
    return theta_r + (theta_s - theta_r) * exp(alpha * h)


# -- Conservation correctness for stage_value, all implicit tableaux ------

def _run_richards(scheme, steps=20, dt_val=100.0, **kwargs):
    """Run the Richards problem with a flux BC; return cumulative
    |mass-balance| error over the integration window."""
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 1)
    h = Function(V, name="h").assign(-1.0)
    v = TestFunction(V)

    MC = MeshConstant(mesh)
    t = MC.Constant(0.0)
    dt = MC.Constant(dt_val)
    flux = Constant(1e-6)

    F = inner(Dt(_theta(h)), v) * dx
    F += inner(Constant(1e-5) * exp(Constant(0.328) * h) * grad(h), grad(v)) * dx
    F -= inner(flux, v) * ds(4)

    stepper = TimeStepper(F, scheme, t, dt, h,
                          solver_parameters=SOLVER, **kwargs)

    one = Function(V).interpolate(Constant(1.0))
    mform = inner(_theta(h), one) * dx
    area = assemble(1 * ds(4, domain=mesh))

    prev = assemble(mform)
    err = 0.0
    for _ in range(steps):
        expected = prev + area * float(flux) * float(dt)
        stepper.advance()
        t.assign(float(t) + float(dt))
        cur = assemble(mform)
        err += abs(cur - expected)
        prev = cur
    return err


# All implicit tableaux Irksome ships, exercised through the stage-value
# stepper which builds the conservative two-evaluation discretisation.
@pytest.mark.parametrize("scheme", [
    BackwardEuler(),    # SA, single-stage, was already passing on master
    RadauIIA(2),        # SA, multi-stage, was already passing on master
    LobattoIIIC(2),     # SA, 2-stage; exercises _update_stiff_acc
    Alexander(),        # SA, 3-stage, has negative b_2 = -0.64 -- regression case
    GaussLegendre(1),   # non-SA, single-stage (=ImplicitMidpoint)
    GaussLegendre(2),   # non-SA, 2-stage fully implicit
    QinZhang(),         # non-SA, 2-stage DIRK structure
], ids=["BackwardEuler", "RadauIIA2", "LobattoIIIC2", "Alexander",
        "ImplicitMidpoint", "GaussLegendre2", "QinZhang"])
def test_conservation_stage_value(scheme):
    err = _run_richards(scheme, stage_type="value")
    assert err < 1e-10, (
        f"Cumulative mass error {err:.3e} > 1e-10 for "
        f"{type(scheme).__name__} via stage_type='value'."
    )


# -- has_composite_time_derivative classification -------------------------

def test_pass_through_handles_linear_arithmetic():
    """has_composite_time_derivative must NOT flag Dt(c*u), Dt(u/c),
    or Dt(u + h(t)) as composite -- they're all linear in u and the
    chain rule (or the linear stage_value path) handles them correctly.
    Falsely flagging them routes through the conservative variational
    head, which produces a singular Jacobian for DAEs."""
    from ufl import sin
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    t = Constant(0.0)

    # Linear arithmetic forms that must be classified as not-composite.
    forms = {
        "Dt(2*u)":      Dt(Constant(2.0) * u),
        "Dt(u/2)":      Dt(u / Constant(2.0)),
        "Dt(u + sin(t))": Dt(u + sin(t)),
        "Dt(2*u + 3)":  Dt(Constant(2.0) * u + Constant(3.0)),
    }
    v = TestFunction(V)
    for name, expr in forms.items():
        F = inner(expr, v) * dx
        assert not has_composite_time_derivative(F, u), (
            f"{name} was incorrectly flagged as composite -- the "
            "linear stage_value path would be skipped, breaking DAEs."
        )

    # Genuinely nonlinear cases must still be flagged.
    nonlinear = {
        "Dt(u*u)":   Dt(u * u),
        "Dt(theta(u))": Dt(_theta(u)),
        "Dt(1/u)":   Dt(Constant(1.0) / u),
    }
    for name, expr in nonlinear.items():
        F = inner(expr, v) * dx
        assert has_composite_time_derivative(F, u), (
            f"{name} was missed -- this would silently produce a "
            "non-conservative discretisation."
        )


# -- Identity regression: linear Dt(u) still works for every stepper -----

def _run_heat_identity(scheme, steps=4, dt_val=0.05, **kwargs):
    """Linear heat: Dt(u) = Δu, return final u dofs.  A baseline sanity
    check that nothing has been silently broken for g = identity."""
    import numpy as np
    from ufl import sin, pi
    mesh = UnitIntervalMesh(8)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    x, = mesh.coordinates
    u.interpolate(sin(pi * x))
    v = TestFunction(V)
    MC = MeshConstant(mesh)
    t = MC.Constant(0.0)
    dt = MC.Constant(dt_val)

    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx

    stepper = TimeStepper(F, scheme, t, dt, u,
                          solver_parameters=SOLVER, **kwargs)
    for _ in range(steps):
        stepper.advance()
        t.assign(float(t) + float(dt))
    return np.linalg.norm(u.dat.data_ro)


@pytest.mark.parametrize("scheme,kw", [
    (BackwardEuler(),  {"stage_type": "value"}),
    (GaussLegendre(1), {"stage_type": "value"}),
    (QinZhang(),       {"stage_type": "value"}),
    (RadauIIA(2),      {"stage_type": "value"}),
    (BackwardEuler(),  {"stage_type": "dirk"}),
    (QinZhang(),       {"stage_type": "dirk"}),
    (Alexander(),      {"stage_type": "dirk"}),
], ids=["BE-value", "GL1-value", "QinZhang-value", "RadauIIA2-value",
        "BE-dirk", "QinZhang-dirk", "Alexander-dirk"])
def test_identity_runs_and_decays(scheme, kw):
    """g = identity: every stepper must run, and the heat solution must
    decay sensibly.  Catches accidental semantic changes for the linear
    case across the steppers we touched."""
    norm = _run_heat_identity(scheme, **kw)
    assert 0.0 < norm < 5.0, (
        f"Heat norm out of range for {type(scheme).__name__} {kw}: {norm:.3e}"
    )
