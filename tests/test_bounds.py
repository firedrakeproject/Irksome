import numpy as np
import pytest
from firedrake import (DirichletBC, Function, FunctionSpace, SpatialCoordinate, TestFunction,
                       TestFunctions, Constant, UnitSquareMesh, diff, div, dx, exp, grad, inner,
                       norm, pi, sin, split, tanh, sqrt, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, project, as_vector)
from irksome import (Dt, GaussLegendre, MeshConstant, RadauIIA, TimeStepper, BoundsConstrainedDirichletBC)
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


def heat(butcher_tableau, basis_type, bounds_type, **kwargs):
    N = 16

    msh = UnitSquareMesh(N, N)
    V = FunctionSpace(msh, "Lagrange", 1)

    MC = MeshConstant(msh)
    dt = MC.Constant(2 / N)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)

    uexact = 0.5 * exp(-t) * (1 + (tanh((0.1 - sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)) / 0.015)))

    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    v = TestFunction(V)
    u_init = Function(V)

    G = inner(u_init - uexact, v) * dx

    nlvp = NonlinearVariationalProblem(G, u_init)
    nlvs = NonlinearVariationalSolver(nlvp, solver_parameters=vi_params)

    lb = Function(V)
    ub = Function(V)

    ub.assign(np.inf)
    lb.assign(0.0)

    nlvs.solve(bounds=(lb, ub))
    u_c = Function(V)
    u_c.assign(u_init)

    v_c = TestFunction(V)

    F_c = (inner(Dt(u_c), v_c) * dx + inner(grad(u_c), grad(v_c)) * dx - inner(rhs, v_c) * dx)

    bounds = (bounds_type, lb, ub)

    bc = BoundsConstrainedDirichletBC(V, uexact, "on_boundary", (lb, ub), solver_parameters=vi_params)

    kwargs_c = {"bounds": bounds,
                "stage_type": "value",
                "basis_type": basis_type,
                "solver_parameters": vi_params
                }

    stepper_c = TimeStepper(F_c, butcher_tableau, t, dt, u_c, bcs=bc, **kwargs_c)

    violations_for_constrained_method = []

    for _ in range(5):
        stepper_c.advance()
        t.assign(float(t) + float(dt))
        min_value_c = min(u_c.dat.data)
        if min_value_c < 0:
            violations_for_constrained_method.append(min_value_c)

    return violations_for_constrained_method


def wave_H1(butcher_tableau):

    msh = UnitSquareMesh(16, 16)

    W = FunctionSpace(msh, "Lagrange", 2)
    W2 = FunctionSpace(msh, "Lagrange", 2)
    Z = W*W2

    x, y = SpatialCoordinate(msh)

    t_crit = np.sqrt(2) / 10.0

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(t_crit / 16)
    Tf = MC.Constant(t_crit)

    u_init = sin(5*pi*x)*sin(5*pi*y)

    uv = Function(Z)

    phi0 = TestFunction(W)

    bc = [DirichletBC(W, Constant(0), "on_boundary")]
    projection_problem = NonlinearVariationalProblem(
        inner(uv.subfunctions[0] - u_init, phi0) * dx,
        uv.subfunctions[0], bcs=bc)

    projection_solver = NonlinearVariationalSolver(projection_problem, solver_parameters=lu_params)
    projection_solver.solve()
    uv_coll_update = uv.copy(deepcopy=True)

    u, v = split(uv)
    u_coll_update, v_coll_update = split(uv_coll_update)

    phi, psi = TestFunctions(Z)

    F = (inner(Dt(u), phi) * dx - inner(v, phi) * dx + inner(Dt(v), psi) * dx + inner(grad(u), grad(psi)) * dx)

    F_coll_update = (inner(Dt(u_coll_update), phi) * dx - inner(v_coll_update, phi) * dx + inner(Dt(v_coll_update), psi) * dx + inner(grad(u_coll_update), grad(psi)) * dx)

    bc = [DirichletBC(Z.sub(0), Constant(0), "on_boundary"), DirichletBC(Z.sub(1), Constant(0), "on_boundary")]

    stepper = TimeStepper(F, butcher_tableau, t, dt, uv, bcs=bc,
                          stage_type='value',
                          basis_type="Lagrange",
                          solver_parameters=lu_params,
                          use_collocation_update=False)

    stepper_coll_update = TimeStepper(F_coll_update, butcher_tableau, t, dt, uv_coll_update, bcs=bc,
                                      stage_type='value',
                                      basis_type="Lagrange",
                                      solver_parameters=lu_params,
                                      use_collocation_update=True)

    while (float(t) < float(Tf)):
        if float(t) + float(dt) > float(Tf):
            dt.assign(float(Tf) - float(t))

        stepper.advance()
        stepper_coll_update.advance()
        t.assign(float(t) + float(dt))

    return norm(uv - uv_coll_update)


def wave_HDiv(butcher_tableau):

    N = 10

    msh = UnitSquareMesh(N, N)
    V = FunctionSpace(msh, "RT", 2)
    W = FunctionSpace(msh, "DG", 1)
    Z = V*W

    x, y = SpatialCoordinate(msh)
    up0 = project(as_vector([0, 0, sin(pi*x)*sin(pi*y)]), Z)
    u0, p0 = split(up0)

    up0_coll_update = up0.copy(deepcopy=True)
    u0_coll, p0_coll = split(up0_coll_update)

    v, w = TestFunctions(Z)
    F = inner(Dt(u0), v)*dx + inner(div(u0), w) * dx + inner(Dt(p0), w)*dx - inner(p0, div(v)) * dx

    F_collocation_update = inner(Dt(u0_coll), v)*dx + inner(div(u0_coll), w) * dx + inner(Dt(p0_coll), w)*dx - inner(p0_coll, div(v)) * dx

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(1.0/N)

    params = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"
              }

    stepper = TimeStepper(F, butcher_tableau, t, dt, up0,
                          solver_parameters=params,
                          stage_type='value',
                          basis_type='Lagrange',
                          use_collocation_update=False
                          )

    stepper_coll_update = TimeStepper(F_collocation_update, butcher_tableau, t, dt, up0_coll_update,
                                      solver_parameters=params,
                                      stage_type='value',
                                      basis_type='Lagrange',
                                      use_collocation_update=True
                                      )

    while (float(t) < 1.0):
        if float(t) + float(dt) > 1.0:
            dt.assign(1.0 - float(t))

        stepper.advance()
        stepper_coll_update.advance()

        t.assign(float(t) + float(dt))

    return norm(up0 - up0_coll_update)


def wave_H1_bounded(butcher_tableau, spatial_basis, temporal_basis, bounds_type):

    msh = UnitSquareMesh(16, 16, name='mesh')

    W = FunctionSpace(msh, spatial_basis, 2)
    W2 = FunctionSpace(msh, spatial_basis, 2)

    Z = W*W2

    x, y = SpatialCoordinate(msh)

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(np.sqrt(2) / 160.0)
    Tf = MC.Constant(np.sqrt(2) / 10.0)

    u_init = sin(5*pi*x)*sin(5*pi*y)

    vi_opts = {
        "snes_type": "vinewtonrsls",
        "snes_max_it": 100,
        "snes_atol": 1.e-8
    }

    uv = Function(Z, name='uv')

    phi0 = TestFunction(W)

    bc = DirichletBC(W, Constant(0), "on_boundary")

    projection_problem = NonlinearVariationalProblem(
        inner(uv.subfunctions[0] - u_init, phi0) * dx,
        uv.subfunctions[0], bcs=bc)

    projection_solver = NonlinearVariationalSolver(
        projection_problem, solver_parameters=vi_opts)

    upper = Function(Z)
    lower = Function(Z)

    upper.subfunctions[1].assign(np.inf)
    lower.subfunctions[1].assign(-np.inf)

    upper.subfunctions[0].assign(1.0)
    lower.subfunctions[0].assign(-1.0)

    projection_solver.solve(bounds=(lower.subfunctions[0], upper.subfunctions[0]))
    u, v = split(uv)

    phi, psi = TestFunctions(Z)
    F = (inner(Dt(u), phi) * dx - inner(v, phi) * dx
         + inner(Dt(v), psi) * dx + inner(grad(u), grad(psi)) * dx)

    bc = [DirichletBC(Z.sub(0), Constant(0), "on_boundary"), DirichletBC(Z.sub(1), Constant(0), "on_boundary")]

    stepper = TimeStepper(F, butcher_tableau, t, dt, uv, bcs=bc,
                          stage_type='value',
                          bounds=(bounds_type, lower, upper),
                          basis_type=temporal_basis,
                          solver_parameters=vi_opts)

    bounds_violations = []

    while (float(t) < float(Tf)):
        if float(t) + float(dt) > float(Tf):
            dt.assign(float(Tf) - float(t))

        stepper.advance()
        t.assign(float(t) + float(dt))

        min_val = min(uv.subfunctions[0].dat.data)
        max_val = max(uv.subfunctions[0].dat.data)

        if min_val < -1.0:
            bounds_violations.append(min_val)
        if max_val > 1.0:
            bounds_violations.append(max_val)

    return bounds_violations


@pytest.mark.parametrize('butcher_tableau', [RadauIIA(i) for i in (1, 2)])
@pytest.mark.parametrize('basis_type', ('Bernstein', 'Lagrange'))
@pytest.mark.parametrize('bounds_type', ("stage", "last_stage"))
def test_heat_bounds(butcher_tableau, basis_type, bounds_type):
    error_list = heat(butcher_tableau, basis_type, bounds_type)
    assert len(error_list) == 0


@pytest.mark.parametrize('butcher_tableau', [RadauIIA(i) for i in (1, 2)])
@pytest.mark.parametrize('spatial_basis', ('Bernstein', 'Lagrange'))
@pytest.mark.parametrize('temporal_basis', ('Bernstein', 'Lagrange'))
@pytest.mark.parametrize('bounds_type', ("stage", "last_stage"))
def test_wave_bounds(butcher_tableau, spatial_basis, temporal_basis, bounds_type):
    error_list = wave_H1_bounded(butcher_tableau, spatial_basis, temporal_basis, bounds_type)
    assert len(error_list) == 0


@pytest.mark.parametrize('butcher_tableau', [GaussLegendre(i) for i in (1, 2, 3)])
def test_wave_H1(butcher_tableau):

    assert wave_H1(butcher_tableau) < 1e-13


@pytest.mark.parametrize('butcher_tableau', [GaussLegendre(i) for i in (1, 2, 3)])
def test_wave_HDiv(butcher_tableau):

    assert wave_HDiv(butcher_tableau) < 1e-13
