from math import isclose

import pytest
from firedrake import *
from irksome import Dt, MeshConstant, TimeStepper, DGDescriptor, RadauIIA

import FIAT


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("basis_type", ["Lagrange", "Bernstein", "spectral", "integral"])
def test_1d_heat_dirichletbc(order, basis_type):
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

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    scheme = DGDescriptor(order, basis_type)

    stepper = TimeStepper(
        F, scheme, t, dt, u, bcs=bc, basis_type=basis_type,
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


@pytest.mark.parametrize("order", [0, 1, 2])
def test_1d_heat_neumannbc(order):
    N = 20
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)
    (x,) = SpatialCoordinate(msh)
    butcher_tableau = RadauIIA(order+1)

    uexact = cos(pi*x)*exp(-(pi**2)*t)
    rhs = Dt(uexact) - div(grad(uexact))
    u_Radau = Function(V)
    u = Function(V)
    u_Radau.interpolate(uexact)
    u.interpolate(uexact)

    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(rhs, v) * dx
    )
    F_Radau = replace(F, {u: u_Radau})

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    ufc_line = FIAT.ufc_simplex(1)
    quadrature = FIAT.quadrature.RadauQuadratureLineRule(ufc_line, order+1)

    scheme = DGDescriptor(order, None, quadrature)
    stepper = TimeStepper(
        F, scheme, t, dt, u, quadrature=quadrature,
        solver_parameters=luparams
    )
    stepper_Radau = TimeStepper(
        F_Radau, butcher_tableau, t, dt, u_Radau, solver_parameters=luparams
    )

    t_end = 1.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        stepper_Radau.advance()
        t.assign(float(t) + float(dt))
        assert (errornorm(u_Radau, u) / norm(u)) < 1.e-10


@pytest.mark.parametrize("order", [1, 2, 3])
def test_1d_heat_homogeneous_dirichletbc(order):
    N = 20
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)
    (x,) = SpatialCoordinate(msh)
    butcher_tableau = RadauIIA(order+1)

    uexact = sin(pi*x)*exp(-(pi**2)*t)
    rhs = Dt(uexact) - div(grad(uexact))
    bcs = DirichletBC(V, uexact, "on_boundary")
    u_Radau = Function(V)
    u = Function(V)
    u_Radau.interpolate(uexact)
    u.interpolate(uexact)

    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(rhs, v) * dx
    )
    F_Radau = replace(F, {u: u_Radau})

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    ufc_line = FIAT.ufc_simplex(1)
    quadrature = FIAT.quadrature.RadauQuadratureLineRule(ufc_line, order+1)

    scheme = DGDescriptor(order, None, quadrature)

    stepper = TimeStepper(
        F, scheme, t, dt, u, bcs=bcs, quadrature=quadrature,
        solver_parameters=luparams
    )
    stepper_Radau = TimeStepper(
        F_Radau, butcher_tableau, t, dt, u_Radau, bcs=bcs, solver_parameters=luparams
    )

    t_end = 1.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        stepper_Radau.advance()
        t.assign(float(t) + float(dt))
        assert (errornorm(u_Radau, u) / norm(u)) < 1.e-10
