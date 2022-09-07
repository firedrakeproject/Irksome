import pytest
from firedrake import *
from math import isclose
from irksome import Alexander, Dt, DIRKTimeStepper, TimeStepper
from ufl.algorithms.ad import expand_derivatives
from ufl import replace


@pytest.mark.parametrize("butcher_tableau", [Alexander()])
def test_1d_heat_dirichletbc(butcher_tableau):
    # Boundary values
    u_0 = Constant(2.0)
    u_1 = Constant(3.0)

    N = 10
    x0 = 0.0
    x1 = 10.0
    msh = IntervalMesh(N, x1)
    V = FunctionSpace(msh, "CG", 1)
    dt = Constant(1.0 / N)
    t = Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    # Method of manufactured solutions copied from Heat equation demo.
    S = Constant(2.0)
    C = Constant(1000.0)
    B = (x - Constant(x0)) * (x - Constant(x1)) / C
    R = (x * x) ** 0.5
    # Note end linear contribution
    uexact = (
        B * atan(t) * (pi / 2.0 - atan(S * (R - t)))
        + u_0
        + ((x - x0) / x1) * (u_1 - u_0)
    )
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))
    u = interpolate(uexact, V)
    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(rhs, v) * dx
    )
    bc = [
        DirichletBC(V, u_1, 2),
        DirichletBC(V, u_0, 1),
    ]

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = DIRKTimeStepper(
        F, butcher_tableau, t, dt, u, bcs=bc, solver_parameters=luparams
    )

    t_end = 2.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        # Check solution and boundary values
        print(errornorm(uexact, u) / norm(uexact))
        # assert norm(u - uexact) / norm(uexact) < 10.0 ** -5
        # assert isclose(u.at(x0), u_0)
        # assert isclose(u.at(x1), u_1)


@pytest.mark.parametrize("butcher_tableau", [Alexander()])
def test_1d_heat_neumannbc(butcher_tableau):
    N = 20
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    dt = Constant(1.0 / N)
    t = Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    uexact = cos(pi*x)*exp(-(pi**2)*t)
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))
    u_dirk = interpolate(uexact, V)
    u = interpolate(uexact, V)
    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(rhs, v) * dx
    )
    Fdirk = replace(F, {u: u_dirk})

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = TimeStepper(
        F, butcher_tableau, t, dt, u, solver_parameters=luparams
    )
    stepperdirk = DIRKTimeStepper(
        Fdirk, butcher_tableau, t, dt, u_dirk, solver_parameters=luparams
    )
        
    t_end = 1.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        stepperdirk.advance()
        t.assign(float(t) + float(dt))
        assert(errornorm(u_dirk, u) / norm(u)) < 1.e-10


@pytest.mark.parametrize("butcher_tableau", [Alexander()])
def test_1d_heat_homogdbc(butcher_tableau):
    N = 20
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    dt = Constant(1.0 / N)
    t = Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    uexact = sin(pi*x)*exp(-(pi**2)*t)
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))
    u_dirk = interpolate(uexact, V)
    u = interpolate(uexact, V)
    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(rhs, v) * dx
    )
    bc = [
        DirichletBC(V, 0.0, 'on_boundary')
    ]

    Fdirk = replace(F, {u: u_dirk})

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = TimeStepper(
        F, butcher_tableau, t, dt, u, bcs = bc, solver_parameters=luparams
    )
    stepperdirk = DIRKTimeStepper(
        Fdirk, butcher_tableau, t, dt, u_dirk, bcs = bc, solver_parameters=luparams
    )
        
    t_end = 1.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        stepperdirk.advance()
        t.assign(float(t) + float(dt))
        assert(errornorm(u_dirk, u) / norm(u)) < 1.e-10


