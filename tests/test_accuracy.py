import numpy as np
import pytest
from firedrake import (DirichletBC, Function, FunctionSpace, SpatialCoordinate,
                       TestFunction, UnitIntervalMesh, cos, diff, div, dx,
                       errornorm, exp, grad, inner, norm, pi, solve)
from irksome import Dt, GalerkinCollocationScheme, MeshConstant, RadauIIA, TimeStepper, StageDerivativeNystromTimeStepper


# test the accuracy of the 1d heat equation using CG elements
# and RadauIIA time integration
def heat(n, deg, scheme, **kwargs):
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
    bcs = DirichletBC(V, uexact, "on_boundary")
    rhs = Dt(uexact) - div(grad(uexact))

    v = TestFunction(V)
    u = Function(V)

    F = (inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
         - inner(rhs, v) * dx)

    # Ritz projection is crucial to observe the right order of convergence
    solve(inner(grad(u - uexact), grad(v)) * dx == 0, u, bcs=bcs,
          solver_parameters=params)

    stepper = TimeStepper(F, scheme, t, dt, u, bcs=bcs,
                          solver_parameters=params,
                          **kwargs)

    while float(t) < 1.0:
        if float(t + dt) > 1.0:
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(t + dt)

    return errornorm(uexact, u) / norm(uexact)


@pytest.mark.parametrize("stage_type", ("deriv", "value"))
@pytest.mark.parametrize(('deg', 'convrate', 'time_stages'),
                         [(1, 1.9, i) for i in (1, 2)]
                         + [(2, 2.9, i) for i in (2, 3)])
def test_heat_eq(deg, convrate, time_stages, stage_type):
    diff = np.array([heat(i, deg, RadauIIA(time_stages), stage_type=stage_type) for i in range(3, 6)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (conv > convrate).all()


@pytest.mark.parametrize("stage_type", ("deriv", "value"))
@pytest.mark.parametrize(('deg', 'convrate', 'time_stages'),
                         [(1, 1.9, i) for i in (1, 2)]
                         + [(2, 2.9, i) for i in (2, 3)])
def test_heat_galerkin_collocation(deg, convrate, time_stages, stage_type):
    diff = np.array([heat(i, deg, GalerkinCollocationScheme(time_stages, quadrature_scheme="radau", stage_type=stage_type)) for i in range(3, 6)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (conv > convrate).all()


@pytest.mark.parametrize(('deg', 'convrate', 'time_stages'),
                         [(2, 2.9, i) for i in (2, 3)]
                         + [(3, 3.9, i) for i in (3, 4)])
def test_heat_bern(deg, convrate, time_stages):
    diff = np.array([heat(i, deg, RadauIIA(time_stages), stage_type="value", basis_type="Bernstein") for i in range(3, 8)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (conv > convrate).all()


def telegraph(n, deg, scheme, **kwargs):
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

    uexact = exp(-t) * cos(pi * x)
    bcs = DirichletBC(V, uexact, "on_boundary")
    rhs = Dt(uexact, 2) + Dt(uexact) - div(grad(uexact))

    v = TestFunction(V)
    u = Function(V).interpolate(uexact)
    ut = Function(V).interpolate(diff(uexact, t))

    F = (inner(Dt(Dt(u) + u), v) * dx + inner(grad(u), grad(v)) * dx
         - inner(rhs, v) * dx)

    stepper = StageDerivativeNystromTimeStepper(F, scheme, t, dt, u, ut,
                                                bcs=bcs, solver_parameters=params,
                                                **kwargs)
    while float(t) < 1.0:
        if float(t + dt) > 1.0:
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(t + dt)

    return errornorm(uexact, u) / norm(uexact)


@pytest.mark.parametrize(('deg', 'convrate', 'time_stages'),
                         [(2, 2.8, i) for i in (2, 3)]
                         + [(3, 3.8, i) for i in (3, 4)])
def test_telegraph_eq(deg, convrate, time_stages):
    diff = np.array([telegraph(i, deg, RadauIIA(time_stages)) for i in range(3, 8)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (conv > convrate).all()
