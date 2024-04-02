import pytest
from firedrake import *
from ufl.algorithms.ad import expand_derivatives
from irksome import Dt, MeshConstant, TimeStepper, RadauIIA, LobattoIIIC


def adapt_scalar_heat(N, butcher_tableau):
    msh = UnitSquareMesh(N, N)

    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)

    V = FunctionSpace(msh, "CG", 1)
    x, y = SpatialCoordinate(msh)
    n = FacetNormal(msh)

    uexact = t*(x+y)
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    u = Function(V)
    u.interpolate(uexact)

    v = TestFunction(V)
    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx - inner(inner(grad(uexact), n), v)*ds(2) - inner(inner(grad(uexact), n), v)*ds(4)

    bc = DirichletBC(V, uexact, {1, 3})

    luparams = {"mat_type": "aij",
                "snes_type": "ksponly",
                "ksp_type": "preonly",
                "pc_type": "lu"}

    stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                          solver_parameters=luparams,
                          stage_type="adapt", tol=1e-2)

    while (float(t) < 1.0):
        stepper.dt_max = 1.0 - float(t)
        stepper.advance()
        t.assign(float(t) + float(dt))

    return norm(u-uexact)


def adapt_vector_heat(N, butcher_tableau):
    msh = UnitSquareMesh(N, N)

    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)

    V = VectorFunctionSpace(msh, "CG", 1)
    x, y = SpatialCoordinate(msh)
    n = FacetNormal(msh)

    uexact_1 = t*(x+y)
    uexact_2 = 2*t*(x-y)
    uexact = as_vector([uexact_1, uexact_2])
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    u = Function(V)
    u.interpolate(uexact)

    v = TestFunction(V)
    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx - inner(inner(grad(uexact_1), n), v[0])*ds(2) - inner(inner(grad(uexact_1), n), v[0])*ds(4)

    bc_1 = DirichletBC(V.sub(0), uexact_1, [1, 3])
    bc_2 = DirichletBC(V.sub(1), uexact_2, "on_boundary")
    bcs = [bc_1, bc_2]

    luparams = {"mat_type": "aij",
                "snes_type": "ksponly",
                "ksp_type": "preonly",
                "pc_type": "lu"}

    stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bcs,
                          solver_parameters=luparams,
                          stage_type="adapt", tol=1e-2)

    while (float(t) < 1.0):
        stepper.dt_max = 1.0 - float(t)
        stepper.advance()
        t.assign(float(t) + float(dt))

    return norm(u-uexact)


def adapt_mixed_heat(N, butcher_tableau):
    msh = UnitSquareMesh(N, N)

    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)

    V = FunctionSpace(msh, "CG", 1)
    Z = VectorFunctionSpace(msh, "CG", 2)
    W = V*Z
    x, y = SpatialCoordinate(msh)
    n = FacetNormal(msh)

    uexact_1 = t*(x+y)
    rhs_1 = expand_derivatives(diff(uexact_1, t)) - div(grad(uexact_1))
    uexact_2 = as_vector([2*t*(x-y), t*t*(x*y-2)])
    rhs_2 = expand_derivatives(diff(uexact_2, t)) - div(grad(uexact_2))

    w = Function(W)
    (u1, u2) = split(w)
    v = TestFunction(W)
    (v1, v2) = split(v)

    F1 = inner(Dt(u1), v1)*dx + inner(grad(u1), grad(v1))*dx - inner(rhs_1, v1)*dx - inner(inner(grad(uexact_1), n), v1)*ds(2) - inner(inner(grad(uexact_1), n), v1)*ds(4)
    F2 = inner(Dt(u2), v2)*dx + inner(grad(u2), grad(v2))*dx - inner(rhs_2, v2)*dx - inner(inner(grad(uexact_2[1]), n), v2[1])*ds(1) - inner(inner(grad(uexact_2[1]), n), v2[1])*ds(3)
    F = F1+F2

    bc_1 = DirichletBC(W.sub(0), uexact_1, [1, 3])
    bc_2 = DirichletBC(W.sub(1).sub(0), uexact_2[0], "on_boundary")
    bc_3 = DirichletBC(W.sub(1).sub(1), uexact_2[1], [2, 4])
    bcs = [bc_1, bc_2, bc_3]

    luparams = {"mat_type": "aij",
                "snes_type": "ksponly",
                "ksp_type": "preonly",
                "pc_type": "lu"}

    stepper = TimeStepper(F, butcher_tableau, t, dt, w, bcs=bcs,
                          solver_parameters=luparams,
                          stage_type="adapt", tol=1e-2)

    while (float(t) < 1.0):
        stepper.dt_max = 1.0 - float(t)
        stepper.advance()
        t.assign(float(t) + float(dt))

    u1, u2 = w.subfunctions
    u1.interpolate(uexact_1)
    u2.interpolate(uexact_2)
    return norm(u1-uexact_1) + norm(u2-uexact_2)


@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
@pytest.mark.parametrize('butcher_tableau', [RadauIIA, LobattoIIIC])
def test_adapt_scalar_heat(N, butcher_tableau):
    error = adapt_scalar_heat(N, butcher_tableau(3))
    assert abs(error) < 1e-10


@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
@pytest.mark.parametrize('butcher_tableau', [RadauIIA, LobattoIIIC])
def test_adapt_vector_heat(N, butcher_tableau):
    error = adapt_vector_heat(N, butcher_tableau(3))
    assert abs(error) < 1e-10


@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
@pytest.mark.parametrize('butcher_tableau', [RadauIIA, LobattoIIIC])
def test_adapt_mixed_heat(N, butcher_tableau):
    error = adapt_mixed_heat(N, butcher_tableau(3))
    assert abs(error) < 1e-10
