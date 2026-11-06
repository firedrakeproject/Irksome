import numpy as np
import pytest
from firedrake import (Constant, DirichletBC, Function, FunctionSpace, SpatialCoordinate,
                       TestFunction, TrialFunction, UnitIntervalMesh, VectorFunctionSpace,
                       assemble, cos, div, dx, errornorm, norm, grad, inner, pi, project, sin, split)
from irksome import Dt, expand_time_derivatives, ExplicitNystromTimeStepper, GaussLegendre, MeshConstant, DIRKNystromTimeStepper, StageDerivativeNystromTimeStepper, WSODIRK, ClassicNystrom4Tableau


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

    u = Function(V).interpolate(uexact)
    ut = Function(V)

    v = TestFunction(V)

    F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx

    bc = DirichletBC(V, uexact, "on_boundary")

    E = 0.5 * inner(ut, ut) * dx + 0.5 * inner(grad(u), grad(u)) * dx

    stepper = StageDerivativeNystromTimeStepper(
        F, butcher_tableau, t, dt, u, ut, bcs=bc, solver_parameters=params, bc_type=bc_type)

    E0 = assemble(E)
    for step in range(N):
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

    u = Function(V).interpolate(uexact)
    ut = Function(V)

    v = TestFunction(V)

    F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx

    bc = DirichletBC(V, uexact, "on_boundary")

    E = 0.5 * inner(ut, ut) * dx + 0.5 * inner(grad(u), grad(u)) * dx

    stepper = DIRKNystromTimeStepper(
        F, butcher_tableau, t, dt, u, ut, bcs=bc, solver_parameters=params, bc_type=bc_type)

    E0 = assemble(E)
    for step in range(5*N):
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

    u = Function(V).interpolate(uexact)
    ut = Function(V)

    v = TestFunction(V)

    F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx

    bc = DirichletBC(V, uexact, "on_boundary")

    E = 0.5 * inner(ut, ut) * dx + 0.5 * inner(grad(u), grad(u)) * dx

    stepper = ExplicitNystromTimeStepper(
        F, tableau, t, dt, u, ut, bcs=bc, solver_parameters=params)

    E0 = assemble(E)
    for step in range(5*N):
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

    # Test LinearVariationalSolver interface
    tau, v = split(TestFunction(Z))
    sigma, u = split(TrialFunction(Z))
    F = inner(Dt(u, 2), v) * dx + inner(div(sigma), v) * dx + inner(u, div(tau)) * dx - inner(sigma, tau)*dx

    sigma, u = split(z)
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


def forced_wave_error(n, stepper, tableau, bc_type):
    msh = UnitIntervalMesh(8)
    V = FunctionSpace(msh, "CG", 2)
    x, = SpatialCoordinate(msh)
    t = Constant(0.0)
    dt = Constant(0.5 / n)
    uexact = cos(2 * t) * x * (1 - x) + sin(t) * x
    rhs = expand_time_derivatives(Dt(uexact, 2), t=t, timedep_coeffs=()) - div(grad(uexact))
    u = Function(V).interpolate(uexact)
    ut = Function(V).interpolate(expand_time_derivatives(Dt(uexact), t=t, timedep_coeffs=()))
    v = TestFunction(V)
    F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx - inner(rhs, v) * dx
    bc = DirichletBC(V, uexact, "on_boundary")
    stepper = stepper(F, tableau, t, dt, u, ut, bcs=bc, bc_type=bc_type)
    for _ in range(n):
        stepper.advance()
        t.assign(float(t) + float(dt))
    return errornorm(Function(V).interpolate(uexact), u)


@pytest.mark.parametrize("stepper,tableau,bc_type", [
    (ExplicitNystromTimeStepper, ClassicNystrom4Tableau(), None),
    (ExplicitNystromTimeStepper, ClassicNystrom4Tableau(), "dDAE"),
    (DIRKNystromTimeStepper, WSODIRK(4, 3, 3), "DAE"),
    (DIRKNystromTimeStepper, WSODIRK(4, 3, 3), "dDAE")])
def test_nystrom_forced_dirichlet(stepper, tableau, bc_type):
    """Explicit Nystrom methods only take bc_type="dDAE", their default: the
    time derivative of the boundary data fixes each stage derivative on the
    boundary, and the residual stays at its own stage time.  The exact
    solution is quadratic in x, so the error is purely temporal."""
    errs = [forced_wave_error(n, stepper, tableau, bc_type) for n in (20, 40)]
    assert np.log2(errs[0] / errs[1]) > 2.6
