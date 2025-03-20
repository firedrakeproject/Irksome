from math import isclose

import pytest
from firedrake import *
from irksome import Dt, MeshConstant, GalerkinTimeStepper
from irksome import TimeStepper, GaussLegendre
from FIAT import make_quadrature, ufc_simplex


def run_1d_heat_dirichletbc(order, basis_type):
    # Boundary values
    u_0 = Constant(2.0)
    u_1 = Constant(3.0)

    N = 10
    x0 = 0.0
    x1 = 10.0
    msh = IntervalMesh(N, x1)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)
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
    rhs = Dt(uexact) - div(grad(uexact))
    u = Function(V)
    u.interpolate(uexact)
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

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}

    stepper = GalerkinTimeStepper(
        F, order, t, dt, u, bcs=bc, basis_type=basis_type,
        solver_parameters=luparams
    )

    t_end = 2.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        # Check solution and boundary values
        assert errornorm(uexact, u) / norm(uexact) < 10.0 ** -3
        assert isclose(u.at(x0), u_0)
        assert isclose(u.at(x1), u_1)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("basis_type", ["Lagrange", "Bernstein", "integral"])
def test_1d_heat_dirichletbc(order, basis_type):
    run_1d_heat_dirichletbc(order, basis_type)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_1d_heat_dirichletbc_iso(order):
    run_1d_heat_dirichletbc(1, f"iso({order})")


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("num_quad_points", [3, 4])
def test_1d_heat_neumannbc(order, num_quad_points):
    N = 20
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)
    (x,) = SpatialCoordinate(msh)
    butcher_tableau = GaussLegendre(order)

    uexact = cos(pi*x)*exp(-(pi**2)*t)
    rhs = Dt(uexact) - div(grad(uexact))
    u_GL = Function(V)
    u = Function(V)
    u_GL.interpolate(uexact)
    u.interpolate(uexact)

    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(rhs, v) * dx
    )
    F_GL = replace(F, {u: u_GL})

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    ufc_line = ufc_simplex(1)
    quadrature = make_quadrature(ufc_line, num_quad_points)

    stepper = GalerkinTimeStepper(
        F, order, t, dt, u, quadrature=quadrature,
        solver_parameters=luparams
    )
    stepper_GL = TimeStepper(
        F_GL, butcher_tableau, t, dt, u_GL, solver_parameters=luparams
    )

    t_end = 1.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        stepper_GL.advance()
        t.assign(float(t) + float(dt))
        assert (errornorm(u_GL, u) / norm(u)) < 1.e-10


@pytest.mark.parametrize("order", [1, 2, 3])
def test_1d_heat_homogeneous_dirichletbc(order):
    N = 20
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)
    (x,) = SpatialCoordinate(msh)
    butcher_tableau = GaussLegendre(order)

    uexact = sin(pi*x)*exp(-(pi**2)*t)
    rhs = Dt(uexact) - div(grad(uexact))
    bcs = DirichletBC(V, uexact, "on_boundary")
    u_GL = Function(V)
    u = Function(V)
    u_GL.interpolate(uexact)
    u.interpolate(uexact)

    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(rhs, v) * dx
    )
    F_GL = replace(F, {u: u_GL})

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = GalerkinTimeStepper(
        F, order, t, dt, u, bcs=bcs,
        solver_parameters=luparams
    )
    stepper_GL = TimeStepper(
        F_GL, butcher_tableau, t, dt, u_GL, bcs=bcs, solver_parameters=luparams
    )

    t_end = 1.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        stepper_GL.advance()
        t.assign(float(t) + float(dt))
        assert (errornorm(u_GL, u) / norm(u)) < 1.e-10


def galerkin_wave(n, deg, alpha, order):
    N = 2**n
    msh = UnitIntervalMesh(N)

    params = {"snes_type": "ksponly",
              "ksp_type": "preonly",
              "mat_type": "aij",
              "pc_type": "lu"}

    V = FunctionSpace(msh, "CG", deg)
    W = FunctionSpace(msh, "DG", deg - 1)
    Z = V * W

    x, = SpatialCoordinate(msh)

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(alpha / N)

    up = project(as_vector([0, sin(pi*x)]), Z)
    u, p = split(up)

    v, w = TestFunctions(Z)

    F = (inner(Dt(u), v)*dx + inner(u.dx(0), w) * dx
         + inner(Dt(p), w)*dx - inner(p, v.dx(0)) * dx)

    E = 0.5 * (inner(u, u)*dx + inner(p, p)*dx)

    stepper = GalerkinTimeStepper(F, order, t, dt, up,
                                  solver_parameters=params)

    energies = []

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        energies.append(assemble(E))

    return np.array(energies)


@pytest.mark.parametrize('N', [2**j for j in range(2, 3)])
@pytest.mark.parametrize('deg', (2,))
@pytest.mark.parametrize('order', (1, 2))
def test_wave_eq_galerkin(deg, N, order):
    energy = galerkin_wave(N, deg, 0.3, order)
    assert np.allclose(energy[1:], energy[:-1])
