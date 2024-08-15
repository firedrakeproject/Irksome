import pytest
from firedrake import *
from ufl.algorithms.ad import expand_derivatives
from irksome import Dt, MeshConstant, TimeStepper, RadauIIA, GaussLegendre
from irksome.tools import AI, IA


def heat_inhomog(N, deg, butcher_tableau, stage_type, splitting):
    msh = UnitSquareMesh(N, N)

    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)

    V = FunctionSpace(msh, "CG", 1)
    x, y = SpatialCoordinate(msh)

    uexact = t*(x+y)
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    u = Function(V)
    u.interpolate(uexact)

    v = TestFunction(V)
    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx

    bc = DirichletBC(V, uexact, "on_boundary")

    luparams = {"mat_type": "aij",
                "snes_type": "ksponly",
                "ksp_type": "preonly",
                "pc_type": "lu"}

    stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                          solver_parameters=luparams,
                          stage_type=stage_type,
                          splitting=splitting)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return norm(u-uexact)


@pytest.mark.parametrize('splitting', (AI, IA))
@pytest.mark.parametrize('stage_type', ("deriv", "value"))
@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
@pytest.mark.parametrize('butcher_tableau', [RadauIIA, GaussLegendre])
@pytest.mark.parametrize('time_stages', (1, 2, 3))
def test_inhomog_bc(N, time_stages, butcher_tableau, stage_type, splitting):
    error = heat_inhomog(N, 2, butcher_tableau(time_stages),
                         stage_type, splitting)
    assert abs(error) < 1e-10
