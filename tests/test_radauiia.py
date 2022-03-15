# import pytest
from firedrake import *
# from math import isclose
from irksome import RadauIIA
from irksome.radauiiastuff import TimeStepper
# from ufl.algorithms.ad import expand_derivatives


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
