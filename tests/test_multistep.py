import pytest
from firedrake import (TestFunction, NonlinearVariationalProblem, NonlinearVariationalSolver,
                       UnitSquareMesh, FunctionSpace, Function, grad, sin, pi, cos, project, 
                       SpatialCoordinate, exp, inner, dx, div, norm, diff, DirichletBC)
from irksome import (Dt, MeshConstant, TimeStepper, MultistepTimeStepper, RadauIIA, GaussLegendre)
from ufl.algorithms import expand_derivatives
import numpy as np

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


def heat(msh, N, spatial_degree):

    V = FunctionSpace(msh, "Bernstein", spatial_degree)

    MC = MeshConstant(msh)
    dt = MC.Constant(1 / N)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)
    uexact = exp(-t) * cos(2 * pi * x) ** 2 * sin(2* pi * y) ** 2
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) 

    bc = DirichletBC(V, uexact, "on_boundary")

    u0 = project(uexact, V, bcs=bc)
    t.assign(float(t) + float(dt))
    u1 = project(uexact, V, bcs=bc)

    u2 = Function(V)
    v = TestFunction(V)

    F_hand = inner(u2, v) * dx - (4.0 / 3.0) * inner(u1, v) * dx + (1.0 / 3.0) * inner(u0, v) * dx - (2.0 / 3.0) * dt * (inner(rhs, v) * dx - inner(grad(u2), grad(v)) * dx)

    stepper_prob = NonlinearVariationalProblem(F_hand, u2, bcs=bc)
    stepper_solver = NonlinearVariationalSolver(stepper_prob, solver_parameters=lu_params)

    for i in range(5):
        t.assign(float(t) + float(dt))
        stepper_solver.solve()
        u0.assign(u1)
        u1.assign(u2)
        
    return u2


def heat_mech(msh, N, spatial_degree):

    V = FunctionSpace(msh, "Bernstein", spatial_degree)

    MC = MeshConstant(msh)
    dt = MC.Constant(1 / N)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)
    uexact = exp(-t) * cos(2 * pi * x) ** 2 * sin(2* pi * y) ** 2
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) 

    bc = DirichletBC(V, uexact, "on_boundary")

    u0 = project(uexact, V, bcs=bc)
    t.assign(float(t) + float(dt))
    u1 = project(uexact, V, bcs=bc)

    u = Function(V)
    v = TestFunction(V)

    F = inner(Dt(u), v) * dx - (inner(rhs, v) * dx - inner(grad(u), grad(v)) * dx)

    ## BDF2
    a = np.array([1.0 / 3.0, -4.0 / 3.0, 1.0])
    b = np.array([0.0, 0.0, 2.0 / 3.0])

    stepper = MultistepTimeStepper(F, t, dt, u, (a, b), bcs=bc)
    stepper.us[0].assign(u0)
    stepper.us[1].assign(u1)

    for i in range(5):
        stepper.advance()
        t.assign(float(t) + float(dt))

    return u


def heat_startup(msh, N, spatial_degree, startup_tableau, startup_dt_div):

    V = FunctionSpace(msh, "Bernstein", spatial_degree)

    MC = MeshConstant(msh)
    dt = MC.Constant(1 / N)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)
    uexact = exp(-t) * cos(2 * pi * x) ** 2 * sin(2* pi * y) ** 2
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) 

    bc = DirichletBC(V, uexact, "on_boundary")

    u0 = project(uexact, V, bcs=bc)

    u = project(uexact, V, bcs=bc)
    v = TestFunction(V)

    F = inner(Dt(u), v) * dx - (inner(rhs, v) * dx - inner(grad(u), grad(v)) * dx)

    startup_stepper = TimeStepper(F, startup_tableau, t, dt, u, bcs=bc)


    dt.assign(dt / startup_dt_div)
    for i in range(0, startup_dt_div):
        startup_stepper.advance()
        t.assign(t + dt)
    dt.assign(dt * startup_dt_div)
    u1 = Function(V).assign(u)

    ## BDF2
    a = np.array([1.0 / 3.0, -4.0 / 3.0, 1.0])
    b = np.array([0.0, 0.0, 2.0 / 3.0])

    stepper = MultistepTimeStepper(F, t, dt, u, (a, b), bcs=bc)
    stepper.us[0].assign(u0)
    stepper.us[1].assign(u1)

    for i in range(5):
        stepper.advance()
        t.assign(float(t) + float(dt))
    print(f'hand: {norm(u)}')
    return u


def heat_mech_startup(msh, N, spatial_degree, startup_tableau, startup_dt_div):

    V = FunctionSpace(msh, "Bernstein", spatial_degree)

    MC = MeshConstant(msh)
    dt = MC.Constant(1 / N)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)
    uexact = exp(-t) * cos(2 * pi * x) ** 2 * sin(2* pi * y) ** 2
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) 

    bc = DirichletBC(V, uexact, "on_boundary")

    u = project(uexact, V, bcs=bc)
    v = TestFunction(V)

    F = inner(Dt(u), v) * dx - (inner(rhs, v) * dx - inner(grad(u), grad(v)) * dx)

    ## BDF2
    a = np.array([1.0 / 3.0, -4.0 / 3.0, 1.0])
    b = np.array([0.0, 0.0, 2.0 / 3.0])

    startup_params = {'tableau': startup_tableau,
                      'dt_div': startup_dt_div
                     }

    t.assign(0.0)
    stepper = MultistepTimeStepper(F, t, dt, u, (a, b), bcs=bc, startup_params=startup_params)

    for i in range(5):
        stepper.advance()
        t.assign(float(t) + float(dt))
    print(f'mech: {norm(u)}')
    return u

def heat_bounds(bounds_flag, startup_bounds_flag, startup_tableau):
    N = 16
    msh = UnitSquareMesh(N, N)

    V = FunctionSpace(msh, "Bernstein", 2)

    MC = MeshConstant(msh)
    dt = MC.Constant(1 / N)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)
    uexact = exp(-t) * cos(2 * pi * x) ** 2 * sin(2* pi * y) ** 2
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) 

    bc = DirichletBC(V, uexact, "on_boundary")

    lower = Function(V).assign(0.0)
    upper = Function(V).assign(np.inf)

    u = Function(V)
    v = TestFunction(V)

    F_proj = inner(u - uexact, v) * dx
    project_IC_prob = NonlinearVariationalProblem(F_proj, u, bcs=bc)
    project_IC = NonlinearVariationalSolver(project_IC_prob, solver_parameters=vi_params)
    project_IC.solve(bounds=(lower, upper))

    F = inner(Dt(u), v) * dx - (inner(rhs, v) * dx - inner(grad(u), grad(v)) * dx)

    ## BDF2
    a = np.array([1.0 / 3.0, -4.0 / 3.0, 1.0])
    b = np.array([0.0, 0.0, 2.0 / 3.0])

    t.assign(0.0)

    if startup_bounds_flag:
        startup_bounds = ('stage', lower, upper)
    else:
        startup_bounds = None

    stepper_kwargs = {'solver_parameters': vi_params,
                      'stage_type': 'value',
                      'basis_type': 'Bernstein', 
                      'bounds': startup_bounds 
                      }

    startup_params = {'tableau': startup_tableau,
                      'dt_div': 2,
                      'stepper_kwargs': stepper_kwargs
                     }
    
    if bounds_flag:
        bounds = (lower, upper)
    else:
        bounds = None

    stepper = MultistepTimeStepper(F, t, dt, u, (a, b), bcs=bc, bounds=bounds, solver_parameters=vi_params, startup_params=startup_params)

    min_init = min(stepper.us[0].dat.data)
    min_step1 = min(stepper.us[1].dat.data)
    stepper.advance()
    t.assign(t + dt)
    min_step2 = min(u.dat.data)

    return (min_init >= 0.0, min_step1 >= 0.0, min_step2 >= 0)

@pytest.mark.parametrize('N', [8, 16, 32])
@pytest.mark.parametrize('spatial_degree', [1, 2, 3])
def test_heat_mech(N, spatial_degree):
    msh = UnitSquareMesh(N, N)
    u1 = heat(msh, N, spatial_degree)
    u2 = heat_mech(msh, N, spatial_degree)
    assert norm(u1 - u2) / norm(u1) < 1e-13


@pytest.mark.parametrize('N', [16, 32])
@pytest.mark.parametrize('spatial_degree', [1, 2])
@pytest.mark.parametrize('startup_tableau', [RadauIIA(1), RadauIIA(2), GaussLegendre(2)])
@pytest.mark.parametrize('startup_dt_div', [1, 8])
def test_heat_startup(N, spatial_degree, startup_tableau, startup_dt_div):
    msh = UnitSquareMesh(N, N)
    u1 = heat_startup(msh, N, spatial_degree, startup_tableau, startup_dt_div)
    u2 = heat_mech_startup(msh, N, spatial_degree, startup_tableau, startup_dt_div)
    assert norm(u1 - u2) / norm(u1) < 1e-13


@pytest.mark.parametrize('bounds_flag', (True, False))
@pytest.mark.parametrize('startup_bounds_flag', (True, False))
@pytest.mark.parametrize('startup_tableau', (RadauIIA(1), RadauIIA(2), GaussLegendre(1)))
def test_heat_bounds(bounds_flag, startup_bounds_flag, startup_tableau):
    tup = heat_bounds(bounds_flag, startup_bounds_flag, startup_tableau)
    assert tup == (True, startup_bounds_flag, bounds_flag)
