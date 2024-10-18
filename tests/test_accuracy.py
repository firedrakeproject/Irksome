import numpy as np
import pytest
from firedrake import (DirichletBC, FunctionSpace, SpatialCoordinate,
                       TestFunction, UnitIntervalMesh, cos, diff, div, dx,
                       errornorm, exp, grad, inner, norm, pi, project)
from irksome import Dt, MeshConstant, RadauIIA, TimeStepper
from ufl.algorithms import expand_derivatives


# test the accuracy of the 1d heat equation using CG elements
# and RadauIIA time integration
def heat(n, deg, time_stages, **kwargs):
    N = 2**n
    msh = UnitIntervalMesh(N)

    params = {"snes_type": "ksponly",
              "ksp_type": "preonly",
              "mat_type": "aij",
              "pc_type": "lu"}

    V = FunctionSpace(msh, "CG", deg)
    x, = SpatialCoordinate(msh)

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(2.0 / N)

    uexact = exp(-t) * cos(pi * x)
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    butcher_tableau = RadauIIA(time_stages)

    u = project(uexact, V)

    v = TestFunction(V)

    F = (inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
         - inner(rhs, v) * dx)

    bc = DirichletBC(V, uexact, "on_boundary")

    stepper = TimeStepper(F, butcher_tableau, t, dt, u,
                          bcs=bc, solver_parameters=params,
                          **kwargs)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return errornorm(uexact, u) / norm(uexact)


@pytest.mark.parametrize("kwargs", ({"stage_type": "deriv"},
                                    {"stage_type": "value"}))
@pytest.mark.parametrize(('deg', 'convrate', 'time_stages'),
                         [(1, 1.78, i) for i in (1, 2)]
                         + [(2, 2.8, i) for i in (2, 3)])
def test_heat_eq(deg, convrate, time_stages, kwargs):
    diff = np.array([heat(i, deg, time_stages, **kwargs) for i in range(3, 6)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (conv > convrate).all()


@pytest.mark.parametrize(('deg', 'convrate', 'time_stages'),
                         [(2, 2.8, i) for i in (2, 3)]
                         + [(3, 3.8, i) for i in (3, 4)])
def test_heat_bern(deg, convrate, time_stages):
    diff = np.array([heat(i, deg, time_stages, **{"stage_type": "value", "basis_type": "Bernstein"}) for i in range(3, 8)])
    print(diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print(conv)
    assert (conv > convrate).all()
