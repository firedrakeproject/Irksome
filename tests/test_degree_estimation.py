import pytest
from irksome import Dt
from firedrake import *

from irksome.estimate_degrees import estimate_time_degree


@pytest.fixture
def mesh():
    return UnitSquareMesh(1, 1)


@pytest.fixture(params=("scalar", "vector"))
def V(request, mesh):
    if request.param == "scalar":
        return FunctionSpace(mesh, "CG", 1)
    elif request.param == "vector":
        return VectorFunctionSpace(mesh, "CG", 1)


def test_time():
    k = 3
    t = Constant(0)
    expr = t
    expected = 1
    assert estimate_time_degree(expr, k-1, k, t=t, timedep_coeffs=()) == expected

    expr = (t-1)**2
    expected = 2
    assert estimate_time_degree(expr, k-1, k, t=t, timedep_coeffs=()) == expected

    expr = Dt((t-1)**2)
    expected = 1
    assert estimate_time_degree(expr, k-1, k, t=t, timedep_coeffs=()) == expected


@pytest.mark.parametrize("order", range(5))
def test_time_derivative(V, order):
    k = 3
    u = Function(V)
    expr = u if order == 0 else Dt(u, order)
    expected = max(k - order, 0)
    time_degree = estimate_time_degree(expr, k-1, k, timedep_coeffs=(u,))
    assert time_degree == expected


def test_grad(V):
    k = 3
    u = Function(V)
    expr = grad(u)

    expected = k
    time_degree = estimate_time_degree(expr, k-1, k, timedep_coeffs=(u,))
    assert time_degree == expected


def test_form(V):
    k = 3
    u = Function(V)
    expr = inner(grad(u), grad(u)) * dx

    expected = 2 * k
    time_degree = estimate_time_degree(expr, k-1, k, timedep_coeffs=(u,))
    assert time_degree == expected


def test_form_Dt(V):
    k = 3
    trial_degree = k
    test_degree = k - 1

    u = Function(V)
    v = TestFunction(V)
    expr = inner(Dt(u), v) * dx

    expected = trial_degree-1 + test_degree
    time_degree = estimate_time_degree(expr, test_degree, trial_degree, timedep_coeffs=(u,))
    assert time_degree == expected


def test_nonlinear_form(V):
    k = 3
    trial_degree = k
    test_degree = k - 1

    u = Function(V)
    v = TestFunction(V)
    if u.ufl_shape == ():
        dim, = grad(u).ufl_shape
        c = as_vector([u] + [0]*(dim-1))
    else:
        c = u
    expr = inner(dot(grad(u), c), v) * dx + inner(Dt(u), v) * dx

    expected = 2 * trial_degree + test_degree
    time_degree = estimate_time_degree(expr, test_degree, trial_degree, timedep_coeffs=(u,))
    assert time_degree == expected


def test_mixed_form(V):
    k = 3
    trial_degree = k
    test_degree = k - 1

    Z = V * V
    z = Function(Z)

    v0, v1 = TestFunctions(Z)
    u0, u1 = split(z)
    expr = inner(Dt(u0), v0) * dx + inner(Dt(u1), v1) * dx

    expected = trial_degree-1 + test_degree
    time_degree = estimate_time_degree(expr, test_degree, trial_degree, timedep_coeffs=(z,))
    assert time_degree == expected
