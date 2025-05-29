import pytest
from firedrake import (Constant, DirichletBC, Function, FunctionSpace, SpatialCoordinate,
                       TestFunction, UnitIntervalMesh, VectorFunctionSpace, assemble, cos, div, dx,
                       norm, grad, inner, pi, project, sin, split)
from irksome import Dt, ExplicitNystromTimeStepper, GaussLegendre, MeshConstant, NystromDIRKTimeStepper, StageDerivativeNystromTimeStepper, WSODIRK, ClassicNystrom4Tableau


def wave(n, deg, time_stages, bc_type):
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
    dt = MC.Constant(1.0 / N)

    uexact = 0.5*(cos(pi * (x - t)) + cos(pi * (x + t)))

    butcher_tableau = GaussLegendre(time_stages)

    u = project(uexact, V)
    ut = Function(V)

    v = TestFunction(V)

    F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx

    bc = DirichletBC(V, uexact, "on_boundary")

    E = 0.5 * inner(ut, ut) * dx + 0.5 * inner(grad(u), grad(u)) * dx

    stepper = StageDerivativeNystromTimeStepper(
        F, butcher_tableau, t, dt, u, ut, bcs=bc, solver_parameters=params, bc_type=bc_type)

    E0 = assemble(E)
    tf = 1
    while (float(t) < tf):
        if (float(t) + float(dt) > tf):
            dt.assign(tf - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return assemble(E) / E0, norm(u - uexact)


def dirk_wave(n, deg, bc_type):
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
    dt = MC.Constant(0.2 / N)

    uexact = 0.5*(cos(pi * (x - t)) + cos(pi * (x + t)))

    butcher_tableau = WSODIRK(6, 4, 3)

    u0 = project(uexact, V)
    u = Function(u0)  # copy
    ut = Function(V)

    v = TestFunction(V)

    F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx

    bc = DirichletBC(V, uexact, "on_boundary")

    E = 0.5 * inner(ut, ut) * dx + 0.5 * inner(grad(u), grad(u)) * dx

    stepper = NystromDIRKTimeStepper(
        F, butcher_tableau, t, dt, u, ut, bcs=bc, solver_parameters=params, bc_type=bc_type)

    E0 = assemble(E)
    tf = 1
    while (float(t) < tf - 1e-12):
        if (float(t) + float(dt) > tf - 1e-12):
            dt.assign(tf - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return assemble(E) / E0, norm(u - uexact)


def explicit_dirk_wave(n, deg):
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
    dt = MC.Constant(0.2 / N)

    uexact = 0.5*(cos(pi * (x - t)) + cos(pi * (x + t)))

    tableau = ClassicNystrom4Tableau()

    u0 = project(uexact, V)
    u = Function(u0)  # copy
    ut = Function(V)

    v = TestFunction(V)

    F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx

    bc = DirichletBC(V, uexact, "on_boundary")

    E = 0.5 * inner(ut, ut) * dx + 0.5 * inner(grad(u), grad(u)) * dx

    stepper = ExplicitNystromTimeStepper(
        F, tableau, t, dt, u, ut, bcs=bc, solver_parameters=params)

    E0 = assemble(E)
    tf = 1
    while (float(t) < tf):
        if (float(t) + float(dt) > tf):
            dt.assign(tf - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return assemble(E) / E0, norm(u - uexact)


@pytest.mark.parametrize("bc_type", ["ODE", "DAE", "dDAE"])
def test_wave_eq(bc_type):
    # number of refinements
    n = 5
    deg = 2
    stage_count = 3
    Erat, diff = wave(n, deg, stage_count, bc_type)
    print(Erat, diff)
    assert abs(Erat - 1) < 1.e-6
    assert diff < 3.e-5


@pytest.mark.parametrize("bc_type", ["DAE", "dDAE"])
def test_dirk_wave_eq(bc_type):
    # number of refinements
    n = 5
    deg = 2
    Erat, diff = dirk_wave(n, deg, bc_type)
    print(Erat, diff)
    assert abs(Erat - 1) < 1.e-3
    assert diff < 3.e-4


def test_explicit_dirk_wave_eq():
    # number of refinements
    n = 5
    deg = 2
    Erat, diff = explicit_wave(n, deg)
    print(Erat, diff)
    assert abs(Erat - 1) < 1.e-6
    assert diff < 3.e-5


def mixed_wave(n, deg, time_stages):
    N = 2**n
    msh = UnitIntervalMesh(N)

    params = {"snes_type": "ksponly",
              "ksp_type": "preonly",
              "mat_type": "aij",
              "pc_type": "lu"}

    V = FunctionSpace(msh, "DG", deg-1)
    S = VectorFunctionSpace(msh, "CG", deg)
    Z = S * V
    x, = SpatialCoordinate(msh)

    t = Constant(0.0)
    dt = Constant(1.0 / N)

    uinit = sin(pi * x)

    butcher_tableau = GaussLegendre(time_stages)

    z0 = Function(Z)
    sigma0, u0 = z0.subfunctions
    u0.project(uinit)
    sigma0.project(-grad(uinit))

    z = Function(z0)  # copy
    zt = Function(Z)

    sigma, u = split(z)
    tau, v = split(TestFunction(Z))

    F = inner(Dt(u, 2), v) * dx + inner(div(sigma), v) * dx + inner(u, div(tau)) * dx - inner(sigma, tau)*dx

    sigmat, ut = split(zt)
    E = 0.5 * inner(ut, ut) * dx + 0.5 * inner(sigma, sigma) * dx

    stepper = StageDerivativeNystromTimeStepper(
        F, butcher_tableau, t, dt, z, zt, solver_parameters=params)

    E0 = assemble(E)
    tf = 1
    while (float(t) < tf):
        if (float(t) + float(dt) > tf):
            dt.assign(tf - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return assemble(E) / E0, norm(u + u0)


def test_mixed_wave_eq():
    # number of refinements
    n = 5
    deg = 2
    stage_count = 2
    Erat, diff = mixed_wave(n, deg, stage_count)
    print(Erat, diff)
    assert abs(Erat - 1) < 1.e-8
    assert diff < 3.e-5


def explicit_wave(n, deg):
    N = 2**n
    msh = UnitIntervalMesh(N)

    params = {"snes_type": "ksponly",
              "ksp_type": "preonly",
              "mat_type": "aij",
              "pc_type": "lu"}

    V = FunctionSpace(msh, "CG", deg)
    x, = SpatialCoordinate(msh)

    t = Constant(0.0)
    dt = Constant(0.1 / N)

    uinit = sin(pi * x)

    nystrom_tableau = ClassicNystrom4Tableau()

    u0 = project(uinit, V)
    u = Function(u0)  # copy
    ut = Function(V)

    v = TestFunction(V)

    F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx

    bc = DirichletBC(V, 0, "on_boundary")

    E = 0.5 * inner(ut, ut) * dx + 0.5 * inner(grad(u), grad(u)) * dx

    stepper = StageDerivativeNystromTimeStepper(
        F, nystrom_tableau, t, dt, u, ut, bcs=bc, solver_parameters=params, bc_type="dDAE")

    E0 = assemble(E)
    tf = 1
    while (float(t) < tf):
        if (float(t) + float(dt) > tf):
            dt.assign(tf - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return assemble(E) / E0, norm(u + u0)


def test_explicit_wave_eq():
    # Mesh size and polynomial degree
    n = 5
    deg = 2
    Erat, diff = explicit_wave(n, deg)
    print(Erat, diff)
    assert abs(Erat - 1) < 1.e-6
    assert diff < 3.e-5
