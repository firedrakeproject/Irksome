from firedrake import (Constant, DirichletBC, FacetNormal, Function, FunctionSpace, SpatialCoordinate, TestFunction, TestFunctions,
                       UnitTriangleMesh, UnitSquareMesh, VectorFunctionSpace, as_vector, div, dot, ds, dx, errornorm, grad,
                       inner, project, split)
from irksome import Dt, GaussLegendre, RadauIIA, TimeStepper
from irksome.tools import is_ode
import pytest
import numpy as np


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


@pytest.mark.parametrize('tableau', [RadauIIA, GaussLegendre])
@pytest.mark.parametrize('temporal_degree', [2, 3])
@pytest.mark.parametrize('stage_type', ["deriv", "value"])
def test_mixed_heat(tableau, temporal_degree, stage_type):

    msh = UnitSquareMesh(4, 4)
    V = FunctionSpace(msh, "RT", 3)
    W = FunctionSpace(msh, "DG", 2)
    Z = V * W

    butcher_tableau = tableau(temporal_degree)

    dt = Constant(0.05)
    t = Constant(0.0)

    x, y = SpatialCoordinate(msh)

    u_exact = (x*(1-x)+y*(1-y))*(1+t)
    q_exact = -grad(u_exact)
    qu_exact = as_vector([q_exact[0], q_exact[1], u_exact])
    f = Dt(u_exact) + div(q_exact)

    qu = project(qu_exact, Z)
    q, u = split(qu)

    v, w = TestFunctions(Z)
    n = FacetNormal(msh)

    F = (inner(Dt(u), w) * dx + inner(div(q), w) * dx - inner(f, w) * dx
         + inner(q, v) * dx - inner(u, div(v)) * dx - inner(u_exact, dot(v, n))*ds)

    bc = DirichletBC(Z.sub(0), q_exact, "on_boundary")

    stepper = TimeStepper(F, butcher_tableau, t, dt, qu, bcs=bc,
                          stage_type=stage_type)

    e_vals = []
    for i in range(2):
        stepper.advance()

        t.assign(float(t)+float(dt))
        e_vals.append(errornorm(qu_exact, qu))

    assert np.allclose(e_vals, 0)
