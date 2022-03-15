import pytest
from firedrake import *
# from math import isclose
from irksome import RadauIIA
from irksome.radauiiastuff import TimeStepper
from ufl.algorithms.ad import expand_derivatives


def test_1d_heat():
    butcher_tableau = RadauIIA(2)
    N = 64
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    dt = Constant(1.0 / N)
    t = Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    uexact = 1 + exp(-4*pi*pi*t) * cos(2 * pi * x)

    u = interpolate(uexact, V)
    v = TestFunction(V)
    # Note, implicit Dt(u) for these methods
    F = (
        inner(grad(u), grad(v)) * dx
    )

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}
    stepper = TimeStepper(
        F, butcher_tableau, t, dt, u, solver_parameters=luparams)

    t_end = 0.1
    while float(t) < t_end:
        print(errornorm(uexact, u))
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        # Check solution and boundary values
        # assert norm(u - uexact) / norm(uexact) < 10.0 ** -5


def heat_inhomog(N, deg, butcher_tableau):
    dt = Constant(1.0 / N)
    t = Constant(0.0)

    msh = UnitSquareMesh(N, N)

    V = FunctionSpace(msh, "CG", 1)
    x, y = SpatialCoordinate(msh)

    uexact = t*(x+y)
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    u = interpolate(uexact, V)

    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - inner(rhs, v)*dx

    bc = DirichletBC(V, uexact, "on_boundary")

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


@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
# @pytest.mark.parametrize(('deg', 'time_stages'),
#                          [(1, i) for i in (1, 2)]
#                          + [(2, i) for i in (2, 3)])
def test_inhomog_bc(N):
    error = heat_inhomog(N, 1, RadauIIA(2))
    assert abs(error) < 1e-10
