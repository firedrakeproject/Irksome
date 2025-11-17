import pytest
from firedrake import (TestFunction, NonlinearVariationalProblem, NonlinearVariationalSolver,
                       UnitSquareMesh, FunctionSpace, Function, grad, sin, pi, cos, project, 
                       SpatialCoordinate, split, TestFunctions, Constant, exp, conditional, 
                       Or, And, inner, dx, div, norm, replace, diff, DirichletBC)
from irksome import (Dt, MeshConstant, TimeStepper, MultistepTimeStepper, MultistepMethod, MultistepTableau, RadauIIA, GaussLegendre)
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

    BDF2 = MultistepMethod('BDF', 2)

    stepper = MultistepTimeStepper(F, t, dt, u, BDF2, bcs=bc)
    stepper.us[0].assign(u0)
    stepper.us[1].assign(u1)

    for i in range(5):
        stepper.advance()
        t.assign(float(t) + float(dt))

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

    BDF2 = MultistepMethod('BDF', 2)

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

    startup_parameters = {'tableau': startup_tableau,
                          'dt_div': 2,
                          'stepper_kwargs': stepper_kwargs
                         }
    
    if bounds_flag:
        bounds = (lower, upper)
    else:
        bounds = None

    stepper = MultistepTimeStepper(F, t, dt, u, BDF2, bcs=bc, bounds=bounds, solver_parameters=vi_params, startup_parameters=startup_parameters)

    min_init = min(stepper.us[0].dat.data)
    min_step1 = min(stepper.us[1].dat.data)
    stepper.advance()
    t.assign(t + dt)
    min_step2 = min(u.dat.data)

    return (min_init >= 0.0, min_step1 >= 0.0, min_step2 >= 0)


def CH_hand(msh, spatial_degree, startup_tableau):
    V = FunctionSpace(msh, "CG", spatial_degree)
    VV = V * V
    c_mu = Function(VV)

    MC = MeshConstant(msh)
    dt = MC.Constant(1e-4)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)

    c_init = conditional(Or(And(And(x > 1/8, x < 1/2),
                                And(y > 1/8, y < 1/2)),
                            And(And(x > 1/2, x < 7/8),
                                And(y > 1/2, y < 7/8))),
                        Constant(1),
                        Constant(-1))

    c_mu.subfunctions[0].interpolate(c_init)


    kappa = Constant(2**(-10))
    M = Constant(1)

    v, w = TestFunctions(VV)
    c, mu = split(c_mu)

    F_DT = (inner(Dt(c), v) * dx
        + M * inner(grad(mu), grad(v)) * dx
        + inner(mu, w) * dx - inner(c * (c**2 - 1), w) * dx
        - kappa * inner(grad(c), grad(w)) * dx)

    c_mu0 = Function(c_mu)

    startup_stepper = TimeStepper(F_DT, startup_tableau, t, dt, c_mu)
    dt_div = 4
    dt.assign(dt / dt_div)
    for i in range(dt_div):
        startup_stepper.advance()
        t.assign(t + dt)

    dt.assign(dt * dt_div)
    c_mu1 = Function(VV).assign(c_mu)

    F_BDF_RHS = - (M * inner(grad(mu), grad(v)) * dx + inner(mu, w) * dx - inner(c * (c**2 - 1), w) * dx - kappa * inner(grad(c), grad(w)) * dx)

    c0, mu0 = split(c_mu0)
    c1, mu1 = split(c_mu1)

    F_BDF = inner(c, v) * dx - (4.0 / 3.0) * inner(c1, v) * dx + (1.0 / 3.0) * inner(c0, v) * dx - (2.0 / 3.0) * dt * F_BDF_RHS

    stepper_prob = NonlinearVariationalProblem(F_BDF, c_mu)
    stepper = NonlinearVariationalSolver(stepper_prob)

    for i in range(5):
        t.assign(t + dt)
        stepper.solve()
        c_mu0.assign(c_mu1)
        c_mu1.assign(c_mu)

    return c_mu


def CH_mech(msh, spatial_degree, startup_tableau):
    V = FunctionSpace(msh, "Lagrange", spatial_degree)
    VV = V * V
    c_mu = Function(VV)

    MC = MeshConstant(msh)
    dt = MC.Constant(1e-4)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)

    c_init = conditional(Or(And(And(x > 1/8, x < 1/2),
                                And(y > 1/8, y < 1/2)),
                            And(And(x > 1/2, x < 7/8),
                                And(y > 1/2, y < 7/8))),
                        Constant(1),
                        Constant(-1))

    c_mu.subfunctions[0].interpolate(c_init)

    kappa = Constant(2**(-10))
    M = Constant(1)

    v, w = TestFunctions(VV)
    c, mu = split(c_mu)

    F_DT = (inner(Dt(c), v) * dx
        + M * inner(grad(mu), grad(v)) * dx
        + inner(mu, w) * dx - inner(c * (c**2 - 1), w) * dx
        - kappa * inner(grad(c), grad(w)) * dx)

    startup_parameters = {'tableau': startup_tableau, 'dt_div': 4}

    BDF2 = MultistepMethod('BDF', 2)
    stepper = MultistepTimeStepper(F_DT, t, dt, c_mu, BDF2, startup_parameters=startup_parameters)

    for i in range(5):
        stepper.advance()
        t.assign(t + dt)

    return c_mu


def heat_AB2_hand(msh, N, spatial_basis):

    dt_in = 0.01 / N ** 2
    V = FunctionSpace(msh, spatial_basis, 2)

    MC = MeshConstant(msh)
    dt = MC.Constant(dt_in)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)
    uexact = exp(-t) * cos(2 * pi * x) ** 2 * sin(2* pi * y) ** 2
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) 

    bc = DirichletBC(V, uexact, "on_boundary")

    u2 = Function(V)
    v = TestFunction(V)

    u2 = project(uexact, V, bcs=bc)
    F = inner(Dt(u2), v) * dx - (inner(rhs, v) * dx - inner(grad(u2), grad(v)) * dx)

    u0 = project(uexact, V, bcs=bc)

    startup_stepper = TimeStepper(F, RadauIIA(1), t, dt, u2, bcs=bc)

    dt_mod = 4
    dt.assign(dt / dt_mod)

    for i in range(0, dt_mod):
        startup_stepper.advance()
        t.assign(t + dt)

    dt.assign(dt * dt_mod)
    u1 = Function(V).assign(u2)

    rhsu1 = replace(rhs, {t: t - 1 * dt})
    rhsu0 = replace(rhs, {t: t - 2 * dt})

    F_AB2 = inner(u2, v) * dx - (inner(u1, v) * dx + dt * ((3.0 / 2.0) * (inner(rhsu1, v) * dx - inner(grad(u1), grad(v)) * dx) + 
                                (- 1.0 / 2.0) * (inner(rhsu0, v) * dx - inner(grad(u0), grad(v)) * dx)))

    stepper_prob = NonlinearVariationalProblem(F_AB2, u2, bcs=bc)
    stepper = NonlinearVariationalSolver(stepper_prob)

    for i in range(10):
        t.assign(t + dt)
        stepper.solve()
        u0.assign(u1)
        u1.assign(u2)
    
    return u2


def heat_AB2_mech(msh, N, spatial_basis):
    
    dt_in = 0.01 / N ** 2
    V = FunctionSpace(msh, spatial_basis, 2)

    MC = MeshConstant(msh)
    dt = MC.Constant(dt_in)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)
    uexact = exp(-t) * cos(2 * pi * x) ** 2 * sin(2* pi * y) ** 2
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) 

    bc = DirichletBC(V, uexact, "on_boundary")

    u2 = Function(V)
    v = TestFunction(V)

    u2 = project(uexact, V, bcs=bc)
    F = inner(Dt(u2), v) * dx - (inner(rhs, v) * dx - inner(grad(u2), grad(v)) * dx)

    startup_parameters = {'tableau': RadauIIA(1), 'dt_div': 4}

    AB2 = MultistepMethod('AB', 2)
    stepper = MultistepTimeStepper(F, t, dt, u2, AB2, bcs=bc, startup_parameters=startup_parameters)

    for i in range(10):
        stepper.advance()
        t.assign(float(t) + float(dt))

    return u2


def heat_Q_hand(msh, N, spatial_basis):

    dt_in = 0.01 / N ** 2
    V = FunctionSpace(msh, spatial_basis, 2)

    MC = MeshConstant(msh)
    dt = MC.Constant(dt_in)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)
    uexact = exp(-t) * cos(2 * pi * x) ** 2 * sin(2* pi * y) ** 2
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) 

    bc = DirichletBC(V, uexact, "on_boundary")
    v = TestFunction(V)

    u5 = project(uexact, V, bcs=bc)
    F = inner(Dt(u5), v) * dx - (inner(rhs, v) * dx - inner(grad(u5), grad(v)) * dx)

    u0 = project(uexact, V, bcs=bc)

    startup_stepper = TimeStepper(F, RadauIIA(1), t, dt, u5, bcs=bc)

    dt_mod = 4
    dt.assign(dt / dt_mod)

    for i in range(0, dt_mod):
        startup_stepper.advance()
        t.assign(t + dt)
    u1 = Function(V).assign(u5)
    for i in range(0, dt_mod):
        startup_stepper.advance()
        t.assign(t + dt)
    u2 = Function(V).assign(u5)
    for i in range(0, dt_mod):
        startup_stepper.advance()
        t.assign(t + dt)
    u3 = Function(V).assign(u5)
    for i in range(0, dt_mod):
        startup_stepper.advance()
        t.assign(t + dt)
    u4 = Function(V).assign(u5)

    dt.assign(dt * dt_mod)

    rhsu4 = replace(rhs, {t: t - 1 * dt})
    rhsu3 = replace(rhs, {t: t - 2 * dt})
    rhsu0 = replace(rhs, {t: t - 5 * dt})

    F_Q = inner(u5, v) * dx - 0.5 * inner(u3, v) * dx - 0.5 * inner(u2, v) * dx - (
        dt * ((3.0 / 4.0) * (inner(rhsu4, v) * dx - inner(grad(u4), grad(v)) * dx) + 
            (3.0 / 4.0) * (inner(rhsu3, v) * dx - inner(grad(u3), grad(v)) * dx) +
            (- 1.0 / 2.0) * (inner(rhsu0, v) * dx - inner(grad(u0), grad(v)) * dx)))

    stepper_prob = NonlinearVariationalProblem(F_Q, u5, bcs=bc)
    stepper = NonlinearVariationalSolver(stepper_prob)

    for i in range(10):
        t.assign(t + dt)
        stepper.solve()
        u0.assign(u1)
        u1.assign(u2)
        u2.assign(u3)
        u3.assign(u4)
        u4.assign(u5)
    
    return u5


def heat_Q_mech(msh, N, spatial_basis):
    
    dt_in = 0.01 / N ** 2
    V = FunctionSpace(msh, spatial_basis, 2)

    MC = MeshConstant(msh)
    dt = MC.Constant(dt_in)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)
    uexact = exp(-t) * cos(2 * pi * x) ** 2 * sin(2* pi * y) ** 2
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) 

    bc = DirichletBC(V, uexact, "on_boundary")

    v = TestFunction(V)
    u = project(uexact, V, bcs=bc)
    F = inner(Dt(u), v) * dx - (inner(rhs, v) * dx - inner(grad(u), grad(v)) * dx)

    a = np.array([0.0,        0.0, -0.5, -0.5,       0.0,       1.0])
    b = np.array([-1.0 / 2.0, 0.0,  0.0,  3.0 / 4.0, 3.0 / 4.0, 0.0])

    method = MultistepTableau(a, b)

    startup_parameters = {'tableau': RadauIIA(1), 'dt_div': 4}

    stepper = MultistepTimeStepper(F, t, dt, u, method, bcs=bc, startup_parameters=startup_parameters)

    for i in range(10):
        stepper.advance()
        t.assign(float(t) + float(dt))

    return u


@pytest.mark.parametrize('N', [8, 16])
@pytest.mark.parametrize('spatial_degree', [1, 2, 3])
def test_heat_mech(N, spatial_degree):
    msh = UnitSquareMesh(N, N)
    u1 = heat(msh, N, spatial_degree)
    u2 = heat_mech(msh, N, spatial_degree)
    assert norm(u1 - u2) / norm(u1) < 1e-13


@pytest.mark.parametrize('bounds_flag', (True, False))
@pytest.mark.parametrize('startup_bounds_flag', (True, False))
@pytest.mark.parametrize('startup_tableau', (RadauIIA(1), RadauIIA(2), GaussLegendre(1)))
def test_heat_bounds(bounds_flag, startup_bounds_flag, startup_tableau):
    tup = heat_bounds(bounds_flag, startup_bounds_flag, startup_tableau)
    assert tup == (True, startup_bounds_flag, bounds_flag)


@pytest.mark.parametrize('N', [4, 16])
@pytest.mark.parametrize('spatial_degree', [1, 2])
@pytest.mark.parametrize('startup_tableau', [RadauIIA(1), GaussLegendre(1), GaussLegendre(2)])
def test_CH(N, spatial_degree, startup_tableau):
    msh = UnitSquareMesh(N, N)
    c_mu_hand = CH_hand(msh, spatial_degree, startup_tableau)
    c_mu_mech = CH_mech(msh, spatial_degree, startup_tableau)
    assert (norm(c_mu_hand.subfunctions[0] - c_mu_mech.subfunctions[0]) / norm(c_mu_hand.subfunctions[0]) < 1e-13 and 
            norm(c_mu_hand.subfunctions[1] - c_mu_mech.subfunctions[1]) / norm(c_mu_hand.subfunctions[1]) < 1e-13)


@pytest.mark.parametrize('N', [8, 16])
@pytest.mark.parametrize('spatial_basis', ["Lagrange", "Bernstein"])
def test_AB2_mech(N, spatial_basis):
    msh = UnitSquareMesh(N, N)
    u1 = heat_AB2_hand(msh, N, spatial_basis)
    u2 = heat_AB2_mech(msh, N, spatial_basis)
    assert norm(u1 - u2) / norm(u1) < 1e-13


@pytest.mark.parametrize('N', [8, 16])
@pytest.mark.parametrize('spatial_basis', ["Lagrange", "Bernstein"])
def test_Q_mech(N, spatial_basis):
    msh = UnitSquareMesh(N, N)
    u1 = heat_Q_hand(msh, N, spatial_basis)
    u2 = heat_Q_mech(msh, N, spatial_basis)
    assert norm(u1 - u2) / norm(u1) < 1e-13
