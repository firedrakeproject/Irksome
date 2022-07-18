from firedrake import (Function, FunctionSpace, TestFunction, TestFunctions,
                       UnitTriangleMesh, VectorFunctionSpace, div, dx, grad,
                       inner, split)
from irksome import Dt
from irksome.tools import is_ode


# tests the is_ode tool to confirm it's determined whether something is an
# ODE (had time derivatives on all functions) or a DAE (is missing some)
def test_foo():
    mesh = UnitTriangleMesh()
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = inner(Dt(u), v) * dx + inner(u, v) * dx

    assert is_ode(F, u)

    VV = V * V
    u = Function(VV)
    v = TestFunction(VV)
    u0, u1 = split(u)
    v0, v1 = split(v)

    F = inner(Dt(u0), v0) * dx + inner(Dt(u1), v1) * dx + inner(u, v) * dx
    assert is_ode(F, u)

    F = inner(Dt(u0), v0) * dx + inner(u, v) * dx
    assert not is_ode(F, u)

    F = inner(grad(Dt(u0)), grad(v0)) * dx + inner(Dt(u1), v1) * dx + inner(u, v) * dx
    assert is_ode(F, u)

    V0 = VectorFunctionSpace(mesh, "CG", 1)
    Z = V0 * V
    up = Function(Z)
    u, p = split(up)
    v, w = TestFunctions(Z)

    F = (inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
         - inner(p, div(v)) * dx + inner(div(u), w) * dx)

    assert not is_ode(F, up)
    assert is_ode(F + inner(Dt(p), w) * dx, up)
