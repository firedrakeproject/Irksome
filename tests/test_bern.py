import numpy as np
import pytest
from firedrake import (DirichletBC, FacetNormal, Function, FunctionSpace,
                       SpatialCoordinate, TestFunction, TestFunctions,
                       UnitIntervalMesh, UnitSquareMesh, assemble, cos, diff,
                       div, dot, ds, dx, errornorm, exp, grad, inner, norm, pi,
                       project, sin, split)
from irksome import Dt, MeshConstant, RadauIIA, TimeStepper
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


def heat(n, deg, butcher_tableau, solver_parameters,
         **kwargs):
    N = 2**n
    msh = UnitIntervalMesh(N)

    V = FunctionSpace(msh, "Bernstein", deg)
    x, = SpatialCoordinate(msh)

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
                          **kwargs)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        print(min(u.dat.data))

    return errornorm(uexact, u) / norm(uexact)


def mixed_heat(n, deg, butcher_tableau, solver_parameters,
               **kwargs):
    N = 2**n
    msh = UnitSquareMesh(N, N)

    V = FunctionSpace(msh, "RT", deg)
    el_type = "Bernstein" if deg > 1 else "DG"
    W = FunctionSpace(msh, el_type, deg-1)

    Z = V * W

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
                          **kwargs)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        print(min(up.subfunctions[1].dat.data))

    u, p = up.subfunctions
    erru = errornorm(uexact, u) / norm(uexact)
    errp = errornorm(pexact, p) / norm(pexact)
    return erru + errp


def mixed_wave(n, deg, butcher_tableau, solver_parameters,
               **kwargs):
    N = 2**n
    msh = UnitSquareMesh(N, N)

    V = FunctionSpace(msh, "RT", deg)
    el_type = "Bernstein" if deg > 1 else "DG"
    W = FunctionSpace(msh, el_type, deg-1)

    Z = V * W

    x, y = SpatialCoordinate(msh)

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(1.0 / N)

    pexact = sin(pi * x) * sin(pi * y) * exp(-t)

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
                          **kwargs)

    E = 0.5 * inner(u, u) * dx + 0.5 * inner(p, p) * dx
    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        print(f"{assemble(E):.4e}, {min(up.subfunctions[1].dat.data):.5e}, {max(up.subfunctions[1].dat.data):.5e}")

    u, p = up.subfunctions
    errp = errornorm(pexact, p) / norm(pexact)
    return errp


@pytest.mark.parametrize('butcher_tableau', [RadauIIA(i) for i in (1, 2)])
def test_mixed_heat_bern(butcher_tableau):
    deg = 1
    kwargs = {"stage_type": "value",
              "basis_type": "Bernstein",
              "solver_parameters": lu_params}
    diff = np.array([mixed_heat(i, deg, butcher_tableau, **kwargs) for i in range(2, 4)])
    print(diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print(conv)
    assert (conv > (deg-0.1)).all()


# @pytest.mark.parametrize('num_stages', (2, 3))
# @pytest.mark.parametrize('bounds_type', ('stage', 'last_stage'))
# @pytest.mark.parametrize('basis_type', ('Bernstein', None))
# def test_mixed_heat_bern_bounds(num_stages, bounds_type, basis_type):
#     deg = 1
#     bounds = (bounds_type, (None, 0), (None, None))
#     kwargs = {"stage_type": "value",
#               "basis_type": basis_type,
#               "bounds": bounds,
#               "solver_parameters": vi_params}
#     diff = np.array([mixed_heat(i, deg, RadauIIA(num_stages), **kwargs) for i in range(3, 5)])
#     print(diff)
#     conv = np.log2(diff[:-1] / diff[1:])
#     print(conv)
#     assert (conv > (deg-0.1)).all()


# @pytest.mark.parametrize('butcher_tableau', [RadauIIA(i) for i in (1, 2, 3)])
# @pytest.mark.parametrize('bounds_type', ('stage', 'last_stage'))
# @pytest.mark.parametrize('basis_type', ('Bernstein', None))
# def test_heat_bern_bounds(butcher_tableau, bounds_type, basis_type):
#     deg = 1
#     bounds = (bounds_type, 0, None)
#     if bounds_type == "time_level":
#         update_solver_parameters = vi_params
#     else:
#         update_solver_parameters = None
#     kwargs = {"stage_type": "value",
#               "basis_type": basis_type,
#               "bounds": bounds,
#               "solver_parameters": vi_params,
#               "update_solver_parameters": update_solver_parameters}
#     diff = np.array([heat(i, deg, butcher_tableau, **kwargs) for i in range(3, 5)])
#     print(diff)
#     conv = np.log2(diff[:-1] / diff[1:])
#     print(conv)
#     assert (conv > (deg+0.8)).all()


# @pytest.mark.parametrize('num_stages', (1, 2, 3))
# @pytest.mark.parametrize('bounds_type', ('stage', 'last_stage'))
# @pytest.mark.parametrize('basis_type', ('Bernstein', None))
# def test_wave_bounds(num_stages, bounds_type, basis_type):
#     deg = 1
#     # bounds = (bounds_type, (-1, None), (1, None))
#     bounds = (bounds_type, (None, None), (None, None))
#     kwargs = {"stage_type": "value",
#               "basis_type": basis_type,
#               "bounds": bounds,
#               "solver_parameters": vi_params}
#     diff = np.array([wave(i, deg, GaussLegendre(num_stages), **kwargs) for i in range(5, 7)])
#     print(diff)
#     conv = np.log2(diff[:-1] / diff[1:])
#     print(conv)
#     assert (conv > (deg-0.1)).all()

# mixed_wave(5, 2, GaussLegendre(3), stage_type="value",
#            basis_type="Bernstein",
#            bounds=("time_level", (None, -1), (None, 1)),
#            solver_parameters=vi_params,
#            update_solver_parameters=vi_params)
