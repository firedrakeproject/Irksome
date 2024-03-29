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
                          stage_type="adapt", tol=1e-2)

    while (float(t) < 1.0):
        stepper.dt_max = 1.0 - float(t)
        stepper.advance()
        t.assign(float(t) + float(dt))

    return norm(u-uexact)


@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
@pytest.mark.parametrize('butcher_tableau', [RadauIIA, LobattoIIIC])
def test_adapt_scalar_heat(N, butcher_tableau):
    error = adapt_scalar_heat(N, butcher_tableau(3))
    assert abs(error) < 1e-10
