import pytest
from firedrake import *
from irksome import GaussLegendre, Dt, TimeStepper
from irksome.getForm import AI, IA
from ufl.algorithms.ad import expand_derivatives


def curlcross(a, b):
    curla = a[1].dx(0) - a[0].dx(1)
    return as_vector([-curla*b[1], curla*b[0]])


def RTCFtest(N, deg, butcher_tableau, splitting=AI):

    msh = UnitSquareMesh(N, N, quadrilateral=True)

    Ve = FiniteElement("RTCF", msh.ufl_cell(), 2)
    V = FunctionSpace(msh, Ve)

    dt = Constant(0.1 / N)
    t = Constant(0.0)

    x, y = SpatialCoordinate(msh)

    uexact = as_vector([t + 2*t*x + 4*t*y, 7*t + 5*t*x + 6*t*y])
    rhs = expand_derivatives(diff(uexact, t)) + grad(div(uexact))

    u = project(uexact, V)

    v = TestFunction(V)
    F = inner(Dt(u), v)*dx + inner(div(u), div(v))*dx - inner(rhs, v)*dx
    bc = DirichletBC(V, uexact, "on_boundary")

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


@pytest.mark.parametrize(('deg', 'N', 'time_stages', 'splitting'),
                         [(1, 2**j, i, splt) for j in range(2, 4)
                          for i in (1, 2) for splt in (AI, IA)]
                         + [(2, 2**j, i, splt) for j in range(2, 4)
                            for i in (2, 3) for splt in (AI, IA)])
def test_RTCF(deg, N, time_stages, splitting):
    error = RTCFtest(N, deg, GaussLegendre(time_stages), splitting)
    assert abs(error) < 1e-10
