import pytest
from firedrake import (DirichletBC, Function, FunctionSpace, SpatialCoordinate,
                       TestFunction, UnitIntervalMesh, dx, errornorm, exp, grad,
                       inner, pi, sin)
from irksome import BDF, Dt, DiscontinuousGalerkinScheme, MeshConstant, RadauIIA, TimeStepper


def heat(method, mixed, **kwargs):
    msh = UnitIntervalMesh(16)
    V = FunctionSpace(msh, "CG", 2)
    x, = SpatialCoordinate(msh)
    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(0.05)
    uex = exp(-(1 + pi**2) * t) * sin(pi * x)
    u = Function(V).interpolate(uex)
    v = TestFunction(V)
    if mixed:
        F = inner(Dt(u) + u, v) * dx + inner(grad(u), grad(v)) * dx
    else:
        F = inner(Dt(u), v) * dx + inner(u, v) * dx + inner(grad(u), grad(v)) * dx
    stepper = TimeStepper(F, method, t, dt, u, bcs=DirichletBC(V, 0, "on_boundary"), **kwargs)
    for _ in range(10):
        stepper.advance()
        t.assign(float(t) + float(dt))
    return errornorm(uex, u)


@pytest.mark.parametrize("method,kwargs", [(RadauIIA(2), {"stage_type": "value"}),
                                           (BDF(2), {}),
                                           (DiscontinuousGalerkinScheme(1), {})])
def test_mixed_time_summand_raises(method, kwargs):
    with pytest.raises(ValueError, match="summands of their own"):
        heat(method, True, **kwargs)


@pytest.mark.parametrize("method,kwargs,tol", [(RadauIIA(2), {"stage_type": "value"}, 1e-4),
                                               (RadauIIA(2), {"stage_type": "deriv"}, 1e-4),
                                               (BDF(2), {}, 2e-3),
                                               (DiscontinuousGalerkinScheme(1), {}, 1e-4)])
def test_split_time_summand(method, kwargs, tol):
    assert heat(method, False, **kwargs) < tol


def test_mixed_time_summand_stage_derivative():
    assert abs(heat(RadauIIA(2), True, stage_type="deriv") - heat(RadauIIA(2), False, stage_type="deriv")) < 1e-12
