import pytest
from firedrake import (TestFunction, NonlinearVariationalProblem, NonlinearVariationalSolver,
                       UnitSquareMesh, FunctionSpace, Function, grad, sin, pi, cos,
                       SpatialCoordinate, split, TestFunctions, Constant, exp, conditional,
                       Or, And, inner, dx, div, norm, replace, diff, DirichletBC)
from irksome import (Dt, MeshConstant, TimeStepper, BDF, AdamsMoulton, MultistepTableau, RadauIIA, GaussLegendre)
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


def heat_problem(msh=None, N=16, spatial_basis='Lagrange'):
    if msh is None:
        msh = UnitSquareMesh(N, N)

    V = FunctionSpace(msh, spatial_basis, 2)

    MC = MeshConstant(msh)
    dt = MC.Constant(1 / N)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)
    uexact = exp(-t) * cos(2 * pi * x) ** 2 * sin(2 * pi * y) ** 2
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))
    bc = [DirichletBC(V, uexact, i) for i in [1, 2, 3, 4]]

    u = Function(V).interpolate(uexact)
    v = TestFunction(V)
    F_semi = inner(Dt(u), v) * dx - (inner(rhs, v) * dx - inner(grad(u), grad(v)) * dx)

    return V, t, dt, v, u, uexact, rhs, bc, F_semi


def heat_bounds_and_startup(bounds_flag, startup_bounds_flag, startup_tableau):
    V, t, dt, v, u, uexact, rhs, bc, F_semi = heat_problem()

    lower = Function(V).assign(0.0)
    upper = Function(V).assign(np.inf)

    project_IC_prob = NonlinearVariationalProblem(inner(u - uexact, v) * dx, u, bcs=bc)
    project_IC = NonlinearVariationalSolver(project_IC_prob, solver_parameters=vi_params)
    project_IC.solve(bounds=(lower, upper))

    BDF2 = BDF(2)
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
                          'num_startup_steps': 2,
                          'stepper_kwargs': stepper_kwargs
                          }

    if bounds_flag:
        bounds = (lower, upper)
    else:
        bounds = None

    stepper = TimeStepper(F_semi, BDF2, t, dt, u, bcs=bc, bounds=bounds, solver_parameters=vi_params, startup_parameters=startup_parameters)
    stepper.startup()
    t.assign(stepper.startup_t)

    min_init = min(stepper.us[0].dat.data)
    min_step1 = min(stepper.us[1].dat.data)
    stepper.advance()
    t.assign(t + dt)
    min_step2 = min(u.dat.data)

    return (min_init >= 0.0, min_step1 >= 0.0, min_step2 >= 0.0)


def heat_cust_hand(msh, N, spatial_basis):
    V, t, dt, v, u, uexact, rhs, bc, F_semi = heat_problem(msh=msh, N=N, spatial_basis=spatial_basis)
    dt.assign(0.01 / N ** 2)

    u0 = Function(V).interpolate(uexact)
    t.assign(t + dt)
    u1 = Function(V).interpolate(uexact)
    t.assign(t + dt)
    u2 = Function(V).interpolate(uexact)
    t.assign(t + dt)
    u3 = Function(V).interpolate(uexact)
    t.assign(t + dt)
    u4 = Function(V).interpolate(uexact)

    rhsu4 = replace(rhs, {t: t - 1 * dt})
    rhsu3 = replace(rhs, {t: t - 2 * dt})
    rhsu0 = replace(rhs, {t: t - 5 * dt})

    F_cust = (inner(u, v) * dx - 0.5 * inner(u3, v) * dx - 0.5 * inner(u2, v) * dx
              - (dt * ((3.0 / 4.0) * (inner(rhsu4, v) * dx - inner(grad(u4), grad(v)) * dx)
                       + (3.0 / 4.0) * (inner(rhsu3, v) * dx - inner(grad(u3), grad(v)) * dx)
                       + (- 1.0 / 2.0) * (inner(rhsu0, v) * dx - inner(grad(u0), grad(v)) * dx))))

    stepper_prob = NonlinearVariationalProblem(F_cust, u, bcs=bc)
    stepper = NonlinearVariationalSolver(stepper_prob)

    for i in range(10):
        t.assign(t + dt)
        stepper.solve()
        u0.assign(u1)
        u1.assign(u2)
        u2.assign(u3)
        u3.assign(u4)
        u4.assign(u)

    return u


def heat_cust_mech(msh, N, spatial_basis):
    V, t, dt, v, u, uexact, rhs, bc, F_semi = heat_problem(msh=msh, N=N, spatial_basis=spatial_basis)

    dt.assign(0.01 / N ** 2)

    a = np.array([0.0, 0.0, -0.5, -0.5, 0.0, 1.0])
    b = np.array([-1.0 / 2.0, 0.0, 0.0, 3.0 / 4.0, 3.0 / 4.0, 0.0])

    method = MultistepTableau(a, b)

    stepper = TimeStepper(F_semi, method, t, dt, u, bcs=bc)
    stepper.us[0].interpolate(uexact)
    t.assign(t + dt)
    stepper.us[1].interpolate(uexact)
    t.assign(t + dt)
    stepper.us[2].interpolate(uexact)
    t.assign(t + dt)
    stepper.us[3].interpolate(uexact)
    t.assign(t + dt)
    stepper.us[4].interpolate(uexact)

    for i in range(10):
        stepper.advance()
        t.assign(float(t) + float(dt))

    return u


def CH_BDF2_hand(msh, spatial_degree, startup_tableau):
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
    num_startup_steps = 4
    dt.assign(dt / num_startup_steps)
    for i in range(num_startup_steps):
        startup_stepper.advance()
        t.assign(t + dt)

    dt.assign(dt * num_startup_steps)
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


def CH_BDF2_mech(msh, spatial_degree, startup_tableau):
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

    startup_parameters = {'tableau': startup_tableau, 'num_startup_steps': 4}

    BDF2 = BDF(2)
    stepper = TimeStepper(F_DT, BDF2, t, dt, c_mu, startup_parameters=startup_parameters)
    stepper.startup()
    t.assign(stepper.startup_t)

    for i in range(5):
        stepper.advance()
        t.assign(t + dt)

    return c_mu


def heat_startup_tableau(startup_tableau):
    V, t, dt, v, u, uexact, rhs, bc, F_semi = heat_problem()

    BDF3 = BDF(3)

    stepper_kwargs = {'solver_parameters': vi_params}

    startup_parameters = {'tableau': startup_tableau,
                          'num_startup_steps': 2,
                          'stepper_kwargs': stepper_kwargs
                          }

    stepper = TimeStepper(F_semi, BDF3, t, dt, u, bcs=bc, solver_parameters=vi_params, startup_parameters=startup_parameters)
    stepper.startup()
    t.assign(stepper.startup_t)

    return


@pytest.mark.parametrize('bounds_flag', (True, False))
@pytest.mark.parametrize('startup_bounds_flag', (True, False))
@pytest.mark.parametrize('startup_tableau', (RadauIIA(1), RadauIIA(2), GaussLegendre(1)))
def test_heat_bounds(bounds_flag, startup_bounds_flag, startup_tableau):
    tup = heat_bounds_and_startup(bounds_flag, startup_bounds_flag, startup_tableau)
    assert tup == (True, startup_bounds_flag, bounds_flag)


@pytest.mark.parametrize('N', [4, 16])
@pytest.mark.parametrize('spatial_degree', [1, 2])
@pytest.mark.parametrize('startup_tableau', [RadauIIA(1), GaussLegendre(1), GaussLegendre(2)])
def test_CH(N, spatial_degree, startup_tableau):
    msh = UnitSquareMesh(N, N)
    c_mu_hand = CH_BDF2_hand(msh, spatial_degree, startup_tableau)
    c_mu_mech = CH_BDF2_mech(msh, spatial_degree, startup_tableau)
    assert (norm(c_mu_hand.subfunctions[0] - c_mu_mech.subfunctions[0]) / norm(c_mu_hand.subfunctions[0]) < 1e-13
            and norm(c_mu_hand.subfunctions[1] - c_mu_mech.subfunctions[1]) / norm(c_mu_hand.subfunctions[1]) < 1e-13)


@pytest.mark.parametrize('N', [8, 16])
@pytest.mark.parametrize('spatial_basis', ["Lagrange", "Bernstein"])
def test_cust_mech(N, spatial_basis):
    msh = UnitSquareMesh(N, N)
    u1 = heat_cust_hand(msh, N, spatial_basis)
    u2 = heat_cust_mech(msh, N, spatial_basis)
    assert norm(u1 - u2) / norm(u1) < 1e-13


@pytest.mark.parametrize('startup_tableau', tuple([AdamsMoulton(i) for i in (0, 1, 2, 3)] + [BDF(i) for i in (1, 2, 3)]))
def test_startup_tableau(startup_tableau):
    if startup_tableau.num_prev_steps != 1:
        with pytest.raises(AssertionError, match="Cannot use a multistep method to start a multistep method"):
            heat_startup_tableau(startup_tableau)
    else:
        heat_startup_tableau(startup_tableau)
