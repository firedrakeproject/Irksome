from firedrake import (
    dx, inner,
    Constant, Function, FunctionSpace, TestFunction, UnitIntervalMesh)
from irksome import Dt, GaussLegendre, TimeStepper


def test_appctx():
    msh = UnitIntervalMesh(2)
    V = FunctionSpace(msh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    F = inner(Dt(u) + u, v) * dx

    bt = GaussLegendre(1)

    t = Constant(0.0)
    dt = Constant(0.1)

    stepper = TimeStepper(F, bt, t, dt, u, appctx={"hello": "world"})
    assert "hello" in stepper.solver._ctx.appctx
