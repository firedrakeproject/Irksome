import pytest
import numpy as np
from firedrake import (diff, div, dx, errornorm, exp, grad,
                       inner, norm, pi, project, sin,
                       Constant, DirichletBC, FunctionSpace,
                       SpatialCoordinate, TestFunction, UnitIntervalMesh)

from irksome import Dt, TimeStepper, GaussLegendre
from irksome.tools import IA
from ufl.algorithms import expand_derivatives


# test the accuracy of the 1d heat equation using CG elements
# and Gauss-Legendre time integration
def heat(n, deg, time_stages, stage_type="deriv", splitting=IA):
    N = 2**n
    msh = UnitIntervalMesh(N)

    params = {"snes_type": "ksponly",
              "ksp_type": "preonly",
              "mat_type": "aij",
              "pc_type": "lu"}

    V = FunctionSpace(msh, "CG", deg)
    x, = SpatialCoordinate(msh)

    t = Constant(0.0)
    dt = Constant(2.0 / N)

    uexact = exp(-t) * sin(pi * x)
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    butcher_tableau = GaussLegendre(time_stages)

    u = project(uexact, V)

    v = TestFunction(V)

    F = (inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
         - inner(rhs, v) * dx)

    bc = DirichletBC(V, Constant(0), "on_boundary")

    stepper = TimeStepper(F, butcher_tableau, t, dt, u,
                          bcs=bc, solver_parameters=params,
                          stage_type=stage_type,
                          splitting=splitting)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return errornorm(uexact, u) / norm(uexact)


@pytest.mark.parametrize("stage_type", ("deriv", "value"))
@pytest.mark.parametrize(('deg', 'convrate', 'time_stages'),
                         [(1, 1.78, i) for i in (1, 2)]
                         + [(2, 2.8, i) for i in (2, 3)])
def test_heat_eq(deg, convrate, stage_type, time_stages):
    diff = np.array([heat(i, deg, time_stages, stage_type) for i in range(3, 6)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (conv > convrate).all()
