# import pytest
from firedrake import *
# from math import isclose
from ufl.algorithms.ad import expand_derivatives


def test_1d_be():
    N = 64
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    dt = Constant(1. / N)
    t = Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    uexact = 1 + exp(-4*pi*pi*t) * cos(2 * pi * x)

    u0 = interpolate(uexact, V)
    u = Function(V)
    v = TestFunction(V)
    # Note, implicit Dt(u) for these methods
    F = (
        inner(u - u0, v) * dx +
        dt * inner(grad(u), grad(v)) * dx
    )

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem,
                                        solver_parameters=luparams)
    t_end = 0.5
    while float(t) < t_end:
        print(errornorm(uexact, u0))
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        solver.solve()
        u0.assign(u)
        t.assign(float(t) + float(dt))
        # Check solution and boundary values
        # assert norm(u - uexact) / norm(uexact) < 10.0 ** -5


def test_1d_be_dirichlet():
    N = 64
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    dt = Constant(1. / N)
    t = Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    uexact = 1 + exp(-4*pi*pi*t) * cos(2 * pi * x)

    u0 = interpolate(uexact, V)
    u = Function(V)
    v = TestFunction(V)
    # Note, implicit Dt(u) for these methods
    F = (
        inner(u - u0, v) * dx +
        dt * inner(grad(u), grad(v)) * dx
    )

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem,
                                        solver_parameters=luparams)
    t_end = 0.5
    while float(t) < t_end:
        print(errornorm(uexact, u0))
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        solver.solve()
        u0.assign(u)
        t.assign(float(t) + float(dt))
        # Check solution and boundary values
        # assert norm(u - uexact) / norm(uexact) < 10.0 ** -5


# we do the UFL by hand for this one.
def test_1d_riia2():
    N = 64
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    dt = Constant(1. / N)
    t = Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    uexact = 1 + exp(-4*pi*pi*t) * cos(2 * pi * x)

    u0 = interpolate(uexact, V)

    VV = V * V

    uu = Function(VV)
    u1, u2 = split(uu)
    v1, v2 = TestFunctions(VV)

    F = (
        inner(u1-u0, v1) * dx + inner(u2-u0, v2) * dx
        + Constant(5/12) * dt * inner(grad(u1), grad(v1)) * dx
        + Constant(-1/12) * dt * inner(grad(u2), grad(v1)) * dx
        + Constant(3/4) * dt * inner(grad(u1), grad(v2)) * dx
        + Constant(1/4) * dt * inner(grad(u2), grad(v2)) * dx
    )

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    problem = NonlinearVariationalProblem(F, uu)
    solver = NonlinearVariationalSolver(problem,
                                        solver_parameters=luparams)
    t_end = 0.1

    u1data, u2data = uu.split()
    while float(t) < t_end:
        print(errornorm(uexact, u0))
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        solver.solve()
        u0.assign(u2data)
        t.assign(float(t) + float(dt))
        # Check solution and boundary values
        # assert norm(u - uexact) / norm(uexact) < 10.0 ** -5


# we do the UFL by hand for this one.
def test_1d_riia2_dirichlet():
    N = 64
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    dt = Constant(1. / N)
    t = Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    uexact = cos(t) * cos(2 * pi * x)
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    u0 = interpolate(uexact, V)

    VV = V * V

    uu = Function(VV)
    u1, u2 = split(uu)
    v1, v2 = TestFunctions(VV)

    rhs1 = replace(rhs, {t: t+Constant(1/3)*dt})
    rhs2 = replace(rhs, {t: t+dt})
    F = (
        inner(u1-u0, v1) * dx + inner(u2-u0, v2) * dx
        + Constant(5/12) * dt
        * (inner(grad(u1), grad(v1)) * dx
           - inner(rhs1, v1) * dx)
        + Constant(-1/12) * dt
        * (inner(grad(u2), grad(v1)) * dx
           - inner(rhs2, v1) * dx)
        + Constant(3/4) * dt
        * (inner(grad(u1), grad(v2)) * dx
           - inner(rhs1, v2) * dx)
        + Constant(1/4) * dt
        * (inner(grad(u2), grad(v2)) * dx
           - inner(rhs2, v2) * dx)
    )

    bcfunc1 = Function(V)
    bcfunc2 = Function(V)
    bcs = [
        DirichletBC(VV.sub(0), bcfunc1, "on_boundary"),
        DirichletBC(VV.sub(1), bcfunc2, "on_boundary")]

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    problem = NonlinearVariationalProblem(F, uu, bcs=bcs)
    solver = NonlinearVariationalSolver(problem,
                                        solver_parameters=luparams)
    t_end = 0.1

    u1data, u2data = uu.split()
    while float(t) < t_end:
        print(errornorm(uexact, u0))
        bcfunc1.interpolate(replace(uexact, {t: t+Constant(1/3)*dt}))
        bcfunc2.interpolate(replace(uexact, {t: t+dt}))
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        solver.solve()
        u0.assign(u2data)
        t.assign(float(t) + float(dt))
        # Check solution and boundary values
        # assert norm(u - uexact) / norm(uexact) < 10.0 ** -5


# test_1d_be()
# test_1d_riia2()
test_1d_riia2_dirichlet()
