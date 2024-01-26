import pytest
from firedrake import *
from firedrake.__future__ import interpolate
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper
from irksome.tools import AI, IA
from ufl.algorithms.ad import expand_derivatives


def curlcross(a, b):
    curla = a[1].dx(0) - a[0].dx(1)
    return as_vector([-curla*b[1], curla*b[0]])


def curltest(N, deg, butcher_tableau, splitting):

    msh = UnitSquareMesh(N, N)

    Ve = FiniteElement("N1curl", msh.ufl_cell(), 2, variant="integral")
    V = FunctionSpace(msh, Ve)

    MC = MeshConstant(msh)
    dt = MC.Constant(0.1 / N)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)

    uexact = as_vector([t + 2*t*x + 4*t*y + 3*t*(y**2) + 2*t*x*y, 7*t + 5*t*x + 6*t*y - 3*t*x*y - 2*t*(x**2)])
    rhs = expand_derivatives(diff(uexact, t)) + curl(curl(uexact))

    u = assemble(interpolate(uexact, V))

    v = TestFunction(V)
    n = FacetNormal(msh)
    F = inner(Dt(u), v)*dx + inner(curl(u), curl(v))*dx - inner(curlcross(uexact, n), v)*ds - inner(rhs, v)*dx
    bc = DirichletBC(V, uexact, [1, 2])

    luparams = {"mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu"}

    stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                          solver_parameters=luparams,
                          splitting=splitting)

    while (float(t) < 0.1):
        if (float(t) + float(dt) > 0.1):
            dt.assign(0.1 - float(t))
        stepper.advance()
        print(float(t))
        t.assign(float(t) + float(dt))

    return norm(u-uexact)


@pytest.mark.parametrize("splitting", (AI, IA))
@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
@pytest.mark.parametrize(('deg', 'time_stages'),
                         [(1, i) for i in (1, 2)]
                         + [(2, i) for i in (2, 3)])
def test_curl(deg, N, time_stages, splitting):
    error = curltest(N, deg, GaussLegendre(time_stages), AI)
    assert abs(error) < 1e-10
