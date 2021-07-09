import pytest
from firedrake import *
from ufl.algorithms.ad import expand_derivatives
from irksome import GaussLegendre, Dt, TimeStepper
from irksome.getForm import AI, IA


def heat_inhomog(N, deg, butcher_tableau, splitting=AI):
    dt = Constant(1.0 / N)
    t = Constant(0.0)

    msh = UnitSquareMesh(N, N)

    V = FunctionSpace(msh, "CG", 1)
    x, y = SpatialCoordinate(msh)

    uexact = t*(x+y)
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    u = interpolate(uexact, V)

    v = TestFunction(V)
    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx

    bc = DirichletBC(V, uexact, "on_boundary")

    luparams = {"mat_type": "aij",
                "snes_type": "ksponly",
                "ksp_type": "preonly",
                "pc_type": "lu"}

    stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                          solver_parameters=luparams,
                          splitting=splitting)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return norm(u-uexact)


@pytest.mark.parametrize(('deg', 'N', 'time_stages', 'splitting'),
                         [(1, 2**j, i, splt) for j in range(2, 4)
                          for i in (1, 2) for splt in (IA, AI)]
                         + [(2, 2**j, i, splt) for j in range(2, 4)
                            for i in (2, 3) for splt in (IA, AI)])
def test_inhomog_bc(deg, N, time_stages, splitting):
    error = heat_inhomog(N, deg, GaussLegendre(time_stages), splitting)
    assert abs(error) < 1e-10
