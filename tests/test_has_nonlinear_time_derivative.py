"""Classification contract for ``has_nonlinear_time_derivative``.

The helper is what routes stage_value between the existing
linear-combination update (``_update_Ainv``) and the new conservative
variational update.  Misclassification has two failure modes:

* False negative on a nonlinear ``Dt(g(u))`` would silently drop the form
  back onto the linear-combination update and lose mass conservation for
  non-stiffly-accurate tableaux.

* False positive on a linear ``Dt(c*u)`` or ``Dt(u + f(t))`` would route
  the form through the conservative variational head, which has no
  algebraic block and breaks DAEs.

Pin both directions of the contract here so a future change to the
walker cannot quietly regress either case.
"""
import pytest
from firedrake import (
    Constant, Function, FunctionSpace, TestFunction,
    UnitIntervalMesh, dx, exp, inner,
)
from ufl import sin

from irksome import Dt
from irksome.ufl.manipulation import has_nonlinear_time_derivative


def _theta(h, theta_r=Constant(0.15), theta_s=Constant(0.45),
           alpha=Constant(0.328)):
    """Exponential soil moisture, the canonical nonlinear g(h) we care about."""
    return theta_r + (theta_s - theta_r) * exp(alpha * h)


@pytest.fixture
def setup():
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "CG", 1)
    return V, Function(V), TestFunction(V), Constant(0.0)


def test_linear_arithmetic_is_not_flagged(setup):
    """Constant-coefficient scalings of u and additive time forcings must
    not be flagged.  These are linear in u; ``_update_Ainv`` is correct
    for them and is also the path that handles DAE structure."""
    V, u, v, t = setup
    forms = {
        "Dt(2*u)": Dt(Constant(2.0) * u),
        "Dt(u/2)": Dt(u / Constant(2.0)),
        "Dt(u + sin(t))": Dt(u + sin(t)),
        "Dt(2*u + 3)": Dt(Constant(2.0) * u + Constant(3.0)),
    }
    for name, expr in forms.items():
        F = inner(expr, v) * dx
        assert not has_nonlinear_time_derivative(F, u), (
            f"{name} was incorrectly flagged as nonlinear -- the linear "
            "stage_value path would be skipped, breaking DAEs."
        )


def test_nonlinear_is_flagged(setup):
    """Genuinely nonlinear g(u) inside Dt must be flagged so that
    stage_value routes through the conservative variational update."""
    V, u, v, t = setup
    nonlinear = {
        "Dt(u*u)": Dt(u * u),
        "Dt(theta(u))": Dt(_theta(u)),
        "Dt(1/u)": Dt(Constant(1.0) / u),
    }
    for name, expr in nonlinear.items():
        F = inner(expr, v) * dx
        assert has_nonlinear_time_derivative(F, u), (
            f"{name} was missed -- this would silently produce a "
            "non-conservative discretisation for non-SA stage_value."
        )
