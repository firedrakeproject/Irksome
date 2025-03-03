import numpy as np
import pytest
from firedrake import (BrokenElement, DirichletBC, FacetNormal, FiniteElement,
                       Function, FunctionSpace, SpatialCoordinate,
                       TestFunction, TestFunctions, UnitIntervalMesh,
                       UnitSquareMesh, cos, diff, div, dot, ds, dx, errornorm,
                       exp, grad, inner, norm, pi, project, sin, split,
                       triangle)
from firedrake.petsc import PETSc
from irksome import Dt, GaussLegendre, MeshConstant, RadauIIA, TimeStepper
from ufl.algorithms import expand_derivatives

lu_params = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "mat_type": "aij",
    "pc_type": "lu"
}

vi_params = {
    "snes_type": "vinewtonrsls",
    "snes_max_it": 300,
    "snes_atol": 1.e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
}


def heat(n, deg, butcher_tableau, solver_parameters, bounds_type, **kwargs):
    N = 2**n
    msh = UnitIntervalMesh(N)

    V = FunctionSpace(msh, "Bernstein", deg)
    x, = SpatialCoordinate(msh)

    if bounds_type is not None:
        lb = Function(V)
        lb.assign(0)
        ub = None
        bounds = (bounds_type, lb, ub)
    else:
        bounds = None

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(2.0 / N)

    uexact = exp(-t) * cos(pi * x)**2
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    u = project(uexact, V)

    v = TestFunction(V)

    F = (inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
         - inner(rhs, v) * dx)

    bc = DirichletBC(V, uexact, "on_boundary")

    stepper = TimeStepper(F, butcher_tableau, t, dt, u,
                          bcs=bc, solver_parameters=solver_parameters,
                          bounds=bounds, **kwargs)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return errornorm(uexact, u) / norm(uexact)


def mixed_heat(n, deg, butcher_tableau, solver_parameters, bounds_type, **kwargs):
    N = 2**n
    msh = UnitSquareMesh(N, N)

    V = FunctionSpace(msh, "RT", deg)
    if deg == 1:
        W = FunctionSpace(msh, "DG", 0)
    else:
        W = FunctionSpace(msh, BrokenElement(FiniteElement("Bernstein", triangle, deg-1)))

    Z = V * W

    if bounds_type is not None:
        lb = Function(Z)
        lb.subfunctions[0].assign(PETSc.NINFINITY)
        lb.subfunctions[1].assign(0)
        bounds = (bounds_type, lb, None)
    else:
        bounds = None

    x, y = SpatialCoordinate(msh)

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(1.0 / N)

    pexact = (cos(pi * x) * cos(pi * y))**2 * exp(-t)
    uexact = -grad(pexact)

    up = Function(Z)
    u, p = split(up)

    v, w = TestFunctions(Z)

    n = FacetNormal(msh)

    rhs = expand_derivatives(diff(pexact, t) + div(uexact))

    F = (inner(Dt(p), w) * dx
         + inner(div(u), w) * dx
         - inner(rhs, w) * dx
         + inner(u, v) * dx
         - inner(p, div(v)) * dx
         + inner(pexact, dot(v, n)) * ds)

    up.subfunctions[1].project(pexact)

    stepper = TimeStepper(F, butcher_tableau, t, dt, up,
                          solver_parameters=solver_parameters,
                          bounds=bounds,
                          **kwargs)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    u, p = up.subfunctions
    erru = errornorm(uexact, u) / norm(uexact)
    errp = errornorm(pexact, p) / norm(pexact)
    return erru + errp


def mixed_wave(n, deg, butcher_tableau, solver_parameters, bounds_type, **kwargs):
    N = 2**n
    msh = UnitSquareMesh(N, N)

    V = FunctionSpace(msh, "RT", deg)
    if deg == 1:
        W = FunctionSpace(msh, "DG", 0)
    else:
        W = FunctionSpace(msh, BrokenElement(FiniteElement("Bernstein", triangle, deg-1)))

    Z = V * W

    if bounds_type is not None:
        lb = Function(Z)
        lb.subfunctions[0].assign(PETSc.NINFINITY)
        lb.subfunctions[1].assign(-1)
        ub = Function(Z)
        ub.subfunctions[0].assign(PETSc.INFINITY)
        ub.subfunctions[1].assign(1)

        bounds = (bounds_type, lb, ub)
    else:
        bounds = None

    x, y = SpatialCoordinate(msh)

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(1.0 / N)

    pexact = sin(pi * x) * sin(pi * y) * cos(np.sqrt(2) * pi * t)

    up = Function(Z)
    u, p = split(up)

    v, w = TestFunctions(Z)

    n = FacetNormal(msh)

    F = (inner(Dt(p), w) * dx
         + inner(div(u), w) * dx
         + inner(Dt(u), v) * dx
         - inner(p, div(v)) * dx)

    up.subfunctions[1].project(pexact)

    stepper = TimeStepper(F, butcher_tableau, t, dt, up,
                          solver_parameters=solver_parameters,
                          bounds=bounds,
                          **kwargs)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    u, p = up.subfunctions
    errp = errornorm(pexact, p) / norm(pexact)
    return errp


@pytest.mark.parametrize('butcher_tableau', [RadauIIA(i) for i in (1, 2, 3)])
@pytest.mark.parametrize('basis_type', ('Bernstein', None))
@pytest.mark.parametrize('bounds_type', ("stage", "last_stage", None))
def test_heat_bern_bounds(butcher_tableau, basis_type, bounds_type):
    deg = 1

    kwargs = {"stage_type": "value",
              "basis_type": basis_type}

    diff = np.array([heat(i, deg, butcher_tableau, solver_parameters=vi_params,
                          bounds_type=bounds_type, **kwargs) for i in range(3, 5)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (conv > (deg+0.8)).all()


@pytest.mark.parametrize('butcher_tableau', [GaussLegendre(i) for i in (1, 2)])
@pytest.mark.parametrize('basis_type', ('Bernstein', None))
@pytest.mark.parametrize('bounds_type', ("stage", "last_stage", None))
def test_mixed_wave_bern_bounds(butcher_tableau, basis_type, bounds_type):
    deg = 2

    kwargs = {"stage_type": "value",
              "basis_type": basis_type}

    diff = np.array([mixed_wave(i, deg, butcher_tableau, solver_parameters=vi_params,
                                bounds_type=bounds_type, **kwargs) for i in range(3, 5)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (conv > (deg-0.1)).all()


@pytest.mark.parametrize('butcher_tableau', [RadauIIA(i) for i in (1, 2)])
@pytest.mark.parametrize('basis_type', ('Bernstein', None))
@pytest.mark.parametrize('bounds_type', ("stage", "last_stage", None))
def test_mixed_heat_bern_bounds(butcher_tableau, basis_type, bounds_type):
    deg = 1

    kwargs = {"stage_type": "value",
              "basis_type": basis_type}

    diff = np.array([mixed_heat(i, deg, butcher_tableau, solver_parameters=vi_params,
                                bounds_type=bounds_type, **kwargs) for i in range(3, 5)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (conv > (deg-0.1)).all()
