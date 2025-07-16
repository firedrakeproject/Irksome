import numpy as np
import pytest
from firedrake import (DirichletBC, Function, FunctionSpace, SpatialCoordinate, TestFunction,
                       TestFunctions, UnitSquareMesh, diff, div, dx, exp, grad, inner,
                       pi, sin, split, tanh, sqrt, NonlinearVariationalProblem,
                       NonlinearVariationalSolver)
from irksome import (Dt, MeshConstant, BoundsConstrainedDirichletBC,
                     GalerkinTimeStepper, DiscontinuousGalerkinTimeStepper)
from ufl.algorithms import expand_derivatives

from FIAT.quadrature import (RadauQuadratureLineRule, GaussLobattoLegendreQuadratureLineRule)
from FIAT import ufc_simplex

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
    "pc_type": "lu"
}


def heat_CG(quad_rule, order, basis_type, bounds_type):

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

    if quad_rule is not None:
        quad_rule = quad_rule(ufc_simplex(1), order+1)

    stepper = GalerkinTimeStepper(F_c, order, t, dt, u_c, quadrature=quad_rule, basis_type=basis_type, bounds=bounds, bcs=bc, solver_parameters=vi_params)

    violations_for_constrained_method = []

    for _ in range(5):
        stepper.advance()
        t += dt
        min_value_c = min(u_c.dat.data)
        print(min_value_c)
        if min_value_c < 0:
            violations_for_constrained_method.append(min_value_c)

    return violations_for_constrained_method


def heat_DG(quad_rule, order, basis_type, bounds_type):

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

    if quad_rule is not None:
        quad_rule = quad_rule(ufc_simplex(1), order+1)

    stepper = DiscontinuousGalerkinTimeStepper(F_c, order, t, dt, u_c, quadrature=quad_rule, basis_type=basis_type, bounds=bounds, bcs=bc, solver_parameters=vi_params)

    violations_for_constrained_method = []

    for _ in range(5):
        stepper.advance()
        t += dt
        min_value_c = min(u_c.dat.data)
        print(min_value_c)
        if min_value_c < 0:
            violations_for_constrained_method.append(min_value_c)

    return violations_for_constrained_method


def wave_H1_CG(quad_rule, order, basis_type, bounds_type):

    msh = UnitSquareMesh(16, 16, name='mesh')
    W = FunctionSpace(msh, "Lagrange", 1)
    Z = W * W

    x, y = SpatialCoordinate(msh)

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(np.sqrt(2) / 160.0)
    Tf = MC.Constant(np.sqrt(2) / 10.0)

    u_init = sin(5*pi*x)*sin(5*pi*y)

    uv = Function(Z, name='uv')

    phi0 = TestFunction(W)

    bc = DirichletBC(W, 0, "on_boundary")

    projection_problem = NonlinearVariationalProblem(
        inner(uv.subfunctions[0] - u_init, phi0) * dx,
        uv.subfunctions[0], bcs=bc)

    projection_solver = NonlinearVariationalSolver(
        projection_problem, solver_parameters=vi_params)

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

    bc = [DirichletBC(Z.sub(0), 0, "on_boundary"), DirichletBC(Z.sub(1), 0, "on_boundary")]

    bounds = (bounds_type, lower, upper)

    if quad_rule is not None:
        quad_rule = quad_rule(ufc_simplex(1), order+1)

    stepper = GalerkinTimeStepper(F, order, t, dt, uv, quadrature=quad_rule, basis_type=basis_type, bounds=bounds, bcs=bc, solver_parameters=vi_params)

    bounds_violations = []

    while (float(t) < float(Tf)):
        if float(t) + float(dt) > float(Tf):
            dt.assign(float(Tf) - float(t))

        stepper.advance()
        t += dt

        min_val = min(uv.subfunctions[0].dat.data)
        max_val = max(uv.subfunctions[0].dat.data)

        if min_val < -1.0:
            bounds_violations.append(min_val)
        if max_val > 1.0:
            bounds_violations.append(max_val)

    return bounds_violations


def wave_H1_DG(quad_rule, order, basis_type, bounds_type):
    msh = UnitSquareMesh(16, 16, name='mesh')
    W = FunctionSpace(msh, "Lagrange", 1)
    Z = W * W

    x, y = SpatialCoordinate(msh)

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(np.sqrt(2) / 160.0)
    Tf = MC.Constant(np.sqrt(2) / 10.0)

    u_init = sin(5*pi*x)*sin(5*pi*y)

    uv = Function(Z, name='uv')

    phi0 = TestFunction(W)

    bc = DirichletBC(W, 0, "on_boundary")

    projection_problem = NonlinearVariationalProblem(
        inner(uv.subfunctions[0] - u_init, phi0) * dx,
        uv.subfunctions[0], bcs=bc)

    projection_solver = NonlinearVariationalSolver(
        projection_problem, solver_parameters=vi_params)

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

    bc = [DirichletBC(Z.sub(0), 0, "on_boundary"), DirichletBC(Z.sub(1), 0, "on_boundary")]

    bounds = (bounds_type, lower, upper)

    if quad_rule is not None:
        quad_rule = quad_rule(ufc_simplex(1), order+1)

    stepper = DiscontinuousGalerkinTimeStepper(F, order, t, dt, uv, quadrature=quad_rule, basis_type=basis_type, bounds=bounds, bcs=bc, solver_parameters=vi_params)

    bounds_violations = []

    while (float(t) < float(Tf)):
        if float(t) + float(dt) > float(Tf):
            dt.assign(float(Tf) - float(t))

        stepper.advance()
        t += dt

        min_val = min(uv.subfunctions[0].dat.data)
        max_val = max(uv.subfunctions[0].dat.data)

        if min_val < -1.0:
            bounds_violations.append(min_val)
        if max_val > 1.0:
            bounds_violations.append(max_val)

    return bounds_violations


@pytest.mark.parametrize('order', (1, 2, 3))
@pytest.mark.parametrize('quad_rule', [None, GaussLobattoLegendreQuadratureLineRule])
@pytest.mark.parametrize('basis_type', ('Bernstein', 'Lagrange'))
@pytest.mark.parametrize('bounds_type', ("galerkin", ))
def test_heat_CG_bounds(quad_rule, order, basis_type, bounds_type):
    error_list = heat_CG(quad_rule, order, basis_type, bounds_type)
    assert len(error_list) == 0


@pytest.mark.parametrize('order', (1, 2, 3))
@pytest.mark.parametrize('quad_rule', [None, RadauQuadratureLineRule])
@pytest.mark.parametrize('basis_type', ('Bernstein', 'Lagrange'))
@pytest.mark.parametrize('bounds_type', ("galerkin", ))
def test_heat_DG_bounds(quad_rule, order, basis_type, bounds_type):
    error_list = heat_DG(quad_rule, order, basis_type, bounds_type)
    assert len(error_list) == 0


@pytest.mark.parametrize('order', (1, 2, 3))
@pytest.mark.parametrize('quad_rule', [None, GaussLobattoLegendreQuadratureLineRule])
@pytest.mark.parametrize('basis_type', ('Bernstein', 'Lagrange'))
@pytest.mark.parametrize('bounds_type', ("galerkin", ))
def test_wave_CG_bounds(quad_rule, order, basis_type, bounds_type):
    error_list = wave_H1_CG(quad_rule, order, basis_type, bounds_type)
    assert len(error_list) == 0


@pytest.mark.parametrize('order', (1, 2, 3))
@pytest.mark.parametrize('quad_rule', [None, RadauQuadratureLineRule])
@pytest.mark.parametrize('basis_type', ('Bernstein', 'Lagrange'))
@pytest.mark.parametrize('bounds_type', ("galerkin", ))
def test_wave_DG_bounds(quad_rule, order, basis_type, bounds_type):
    error_list = wave_H1_DG(quad_rule, order, basis_type, bounds_type)
    assert len(error_list) == 0
