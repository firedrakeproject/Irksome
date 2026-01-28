import pytest
from firedrake import *
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper
from irksome.tools import AI, IA


def heat_subdomainbc(N, deg, butcher_tableau, splitting=AI):
    msh = UnitSquareMesh(N, N)

    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)

    V = FunctionSpace(msh, "CG", deg)
    x, y = SpatialCoordinate(msh)

    uexact = t*(x+y)
    rhs = Dt(uexact) - div(grad(uexact))

    u = Function(V)
    u.interpolate(uexact)

    v = TestFunction(V)
    n = FacetNormal(msh)
    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx - inner(dot(grad(uexact), n), v)*ds

    bc = DirichletBC(V, uexact, [1, 2])

    luparams = {"mat_type": "aij",
                "snes_type": "ksponly",
                "ksp_type": "preonly",
                "pc_type": "lu"}

    stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                          solver_parameters=luparams)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return norm(u-uexact)


def heat_componentbc(N, deg, butcher_tableau, splitting=AI):
    msh = UnitIntervalMesh(N)

    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)

    V = VectorFunctionSpace(msh, "CG", deg, dim=2)
    (x,) = SpatialCoordinate(msh)

    uexact = as_vector([(x**2-2*x)*t, (1-x**2)*t])
    rhs = Dt(uexact) - div(grad(uexact))

    u = Function(V)
    u.interpolate(uexact)

    v = TestFunction(V)
    n = FacetNormal(msh)
    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx - inner(dot(grad(uexact), n), v)*ds

    bc = [
        DirichletBC(V.sub(0), 0.0, [1]),
        DirichletBC(V.sub(1), 0.0, [2])
    ]

    luparams = {"mat_type": "aij",
                "snes_type": "ksponly",
                "ksp_type": "preonly",
                "pc_type": "lu"}

    stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                          solver_parameters=luparams)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return norm(u-uexact)


@pytest.mark.parametrize('splitting', (AI, IA))
@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
@pytest.mark.parametrize(('deg', 'time_stages'),
                         [(1, i) for i in (1, 2)]
                         + [(2, i) for i in (2, 3)])
def test_subdomainbc(deg, N, time_stages, splitting):
    error = heat_subdomainbc(N, deg, GaussLegendre(time_stages), splitting)
    assert abs(error) < 1e-10


@pytest.mark.parametrize('splitting', (AI, IA))
@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
@pytest.mark.parametrize(('deg', 'time_stages'),
                         [(2, i) for i in (1, 2, 3)])
def test_compbc(deg, N, time_stages, splitting):
    error = heat_componentbc(N, deg, GaussLegendre(time_stages), splitting)
    assert abs(error) < 1e-10
