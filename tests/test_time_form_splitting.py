import pytest
from irksome import Dt
from irksome.manipulation import check_integrals, extract_terms
from ufl import (Coefficient, FiniteElement, FunctionSpace, Mesh, MixedElement,
                 TestFunction, VectorElement, dx, grad, inner, sin, triangle)
from ufl.algorithms.domain_analysis import group_form_integrals


def sig(form):
    form = group_form_integrals(form, form.ufl_domains(),
                                do_append_everywhere_integrals=True)
    return form.signature()


@pytest.fixture
def mesh():
    return Mesh(VectorElement("P", triangle, 1))


@pytest.fixture
def V(mesh):
    return FunctionSpace(mesh, VectorElement("DP", triangle, 1))


@pytest.fixture
def Q(mesh):
    return FunctionSpace(mesh, FiniteElement("RT", triangle, 1))


@pytest.fixture
def W(V, Q):
    mesh = V.ufl_domain()
    return FunctionSpace(mesh, MixedElement(V.ufl_element(), Q.ufl_element()))


def test_can_split(V):
    u = Coefficient(V)

    v = TestFunction(V)

    c = Coefficient(V)

    F = (inner(c, c)*inner(Dt(u), v) + inner(grad(u), grad(v))
         + inner(c, v) + inner(Dt(u), v))*dx

    split = extract_terms(F)

    expect_t = (inner(c, c)*inner(Dt(u), v) + inner(Dt(u), v))*dx
    expect_no_t = inner(grad(u), grad(v))*dx + inner(c, v)*dx

    assert sig(expect_t) == sig(split.time)
    assert sig(expect_no_t) == sig(split.remainder)


def test_can_split_mixed(W):
    u = Coefficient(W)

    v = TestFunction(W)

    c = Coefficient(W)

    F = (inner(c, c)*inner(Dt(u), v) + inner(grad(u), grad(v))
         + inner(c, v) + inner(Dt(u), v))*dx

    split = extract_terms(F)

    expect_t = (inner(c, c)*inner(Dt(u), v) + inner(Dt(u), v))*dx
    expect_no_t = inner(grad(u), grad(v))*dx + inner(c, v)*dx

    assert sig(expect_t) == sig(split.time)
    assert sig(expect_no_t) == sig(split.remainder)

    
def test_can_split_mixed_split(W):
    u = Coefficient(W)
    from ufl import split as splt
    u0, u1 = splt(u)

    v = TestFunction(W)
    v0, v1 = splt(v)

    c = Coefficient(W)

    F = (inner(c, c)*inner(Dt(u0), v0) + inner(grad(u), grad(v))
         + inner(c, v) + inner(Dt(u), v))*dx

    split = extract_terms(F)

    expect_t = (inner(c, c)*inner(Dt(u0), v0) + inner(Dt(u), v))*dx
    expect_no_t = inner(grad(u), grad(v))*dx + inner(c, v)*dx

    assert sig(expect_t) == sig(split.time)
    assert sig(expect_no_t) == sig(split.remainder)


def test_only_first_order(V):
    u = Coefficient(V)

    v = TestFunction(V)

    c = Coefficient(V)

    F = (inner(c, c)*inner(Dt(Dt(u)), v)
         + inner(grad(u), grad(v))
         + inner(c, v)
         + inner(Dt(u), v))*dx

    with pytest.raises(ValueError):
        check_integrals(F.integrals(), expect_time_derivative=True)


@pytest.mark.parametrize("typ", ["mul", "div", "sin"])
def test_Dt_linear(V, typ):
    u = Coefficient(V)

    v = TestFunction(V)

    c = Coefficient(V)

    F = (inner(grad(u), grad(v)) + inner(c, v) + inner(Dt(u), v))*dx

    if typ == "mul":
        F += inner(Dt(u), c)*inner(Dt(u), v)*dx
    elif typ == "div":
        F += inner(Dt(u), v)/Dt(u)[0]*dx
    elif typ == "sin":
        F += inner(sin(Dt(u)[0]), v[0])*dx

    with pytest.raises(ValueError):
        check_integrals(F.integrals(), expect_time_derivative=True)


def test_expecting_time_derivative(V):
    u = Coefficient(V)

    v = TestFunction(V)

    c = Coefficient(V)

    F = (inner(grad(u), grad(v)) + inner(c, v))*dx

    with pytest.raises(ValueError):
        check_integrals(F.integrals(), expect_time_derivative=True)

    check_integrals(F.integrals(), expect_time_derivative=False)

    F += inner(Dt(u), v)*dx
    with pytest.raises(ValueError):
        check_integrals(F.integrals(), expect_time_derivative=False)
