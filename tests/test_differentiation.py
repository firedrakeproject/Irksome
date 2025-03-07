import pytest
from ufl.algorithms import expand_derivatives
from irksome import MeshConstant, Dt, expand_time_derivatives
from firedrake import Constant, diff, dot, FunctionSpace, Function, sin, UnitIntervalMesh, VectorFunctionSpace


@pytest.fixture
def mesh():
    return UnitIntervalMesh(1)


@pytest.fixture(params=("scalar",))
def V(request, mesh):
    if request.param == "scalar":
        return FunctionSpace(mesh, "DG", 0)
    elif request.param == "vector":
        return VectorFunctionSpace(mesh, "DG", 0)


def test_second_derivative(V):
    u = Function(V)
    assert Dt(u, 2) == Dt(Dt(u))


def test_diff(mesh):
    MC = MeshConstant(mesh)
    t = MC.Constant(0.0)
    q = sin(t**2)
    expr = Dt(q)
    expected = expand_derivatives(diff(q, t))
    expr = expand_time_derivatives(expr, t=t)
    assert expr == expected


def test_nested_derivative_expression(mesh):
    MC = MeshConstant(mesh)
    t = MC.Constant(0.0)
    q = sin(t**2)

    expr = Dt(Dt(q) + q)
    expr = expand_time_derivatives(expr, t=t)

    expected = diff(diff(q, t) + q, t)
    expected = expand_derivatives(expected)
    assert expr == expected


def test_expand_sum(V):
    u = Function(V)
    w = Function(V)
    k1 = Constant(1)
    k2 = Constant(2)
    expr = Dt(k1*u + k2*w)

    expr = expand_time_derivatives(expr)
    expected = k1*Dt(u) + k2*Dt(w)
    assert expr == expand_time_derivatives(expected)


def test_expand_product_rule(V):
    u = Function(V)
    w = Function(V)
    expr = Dt(dot(u, w))

    expr = expand_time_derivatives(expr)
    expected = dot(u, Dt(w)) + dot(Dt(u), w)
    assert expr == expand_time_derivatives(expected)


def test_expand_second_derivative_product_rule(V):
    u = Function(V)
    w = Function(V)
    expr = Dt(Dt(dot(u, w)))

    expr = expand_time_derivatives(expr)
    expected = (dot(Dt(u, 2), w)
                + dot(Dt(u), Dt(w))
                + dot(Dt(u), Dt(w))
                + dot(u, Dt(w, 2)))
    # UFL equality is failing here due to different index numbers
    assert str(expr) == str(expected)
