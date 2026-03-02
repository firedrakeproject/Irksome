from math import isclose
import pytest
from firedrake import *
from irksome import WSODIRK, Alexander, Dt, MeshConstant, TimeStepper
from ufl import replace

wsodirks = [WSODIRK(*x) for x in ((4, 3, 2), (4, 3, 3))]


@pytest.mark.parametrize("butcher_tableau", [Alexander()] + wsodirks)
def test_1d_heat_dirichletbc(butcher_tableau):
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

    stepper = TimeStepper(
        F, butcher_tableau, t, dt, u, bcs=bc,
        solver_parameters=luparams,
        stage_type="dirk"
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


@pytest.mark.parametrize("butcher_tableau", [Alexander()] + wsodirks)
def test_1d_heat_neumannbc(butcher_tableau):
    N = 20
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    uexact = cos(pi*x)*exp(-(pi**2)*t)
    rhs = Dt(uexact) - div(grad(uexact))
    u_dirk = Function(V)
    u = Function(V)
    u_dirk.interpolate(uexact)
    u.interpolate(uexact)

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
    stepperdirk = TimeStepper(
        Fdirk, butcher_tableau, t, dt, u_dirk, solver_parameters=luparams,
        stage_type="dirk"
    )

    t_end = 1.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        stepperdirk.advance()
        t.assign(float(t) + float(dt))
        assert (errornorm(u_dirk, u) / norm(u)) < 1.e-10


@pytest.mark.parametrize("butcher_tableau", [Alexander()] + wsodirks)
def test_1d_heat_homogdbc(butcher_tableau):
    N = 20
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    uexact = sin(pi*x)*exp(-(pi**2)*t)
    rhs = Dt(uexact) - div(grad(uexact))
    u_dirk = Function(V)
    u = Function(V)
    u_dirk.interpolate(uexact)
    u.interpolate(uexact)
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
        F, butcher_tableau, t, dt, u, bcs=bc, solver_parameters=luparams
    )
    stepperdirk = TimeStepper(
        Fdirk, butcher_tableau, t, dt, u_dirk, bcs=bc,
        solver_parameters=luparams, stage_type="dirk"
    )

    t_end = 1.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        stepperdirk.advance()
        t.assign(float(t) + float(dt))
        assert (errornorm(u_dirk, u) / norm(u)) < 1.e-10


@pytest.mark.parametrize("butcher_tableau", [Alexander()] + wsodirks)
def test_1d_vectorheat_componentBC(butcher_tableau):
    N = 20
    msh = UnitIntervalMesh(N)
    V = VectorFunctionSpace(msh, "CG", 1, dim=2)
    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    uexact = as_vector([sin(pi*x/2)*exp(-(pi**2)*t/4),
                        cos(pi*x/2)*exp(-(pi**2)*t/4)])
    rhs = Dt(uexact) - div(grad(uexact))
    u_dirk = Function(V)
    u = Function(V)
    u_dirk.interpolate(uexact)
    u.interpolate(uexact)
    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(rhs, v) * dx
    )

    bc = [
        DirichletBC(V.sub(0), 0.0, [1]),
        DirichletBC(V.sub(1), 0.0, [2])
    ]

    Fdirk = replace(F, {u: u_dirk})

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = TimeStepper(
        F, butcher_tableau, t, dt, u, bcs=bc, solver_parameters=luparams
    )
    stepperdirk = TimeStepper(
        Fdirk, butcher_tableau, t, dt, u_dirk, bcs=bc,
        solver_parameters=luparams, stage_type="dirk"
    )

    t_end = 1.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        stepperdirk.advance()
        t.assign(float(t) + float(dt))
        assert (errornorm(u_dirk, u) / norm(u)) < 1.e-10


def subspace_bc(Z, uexact):
    return [DirichletBC(Z.sub(0), uexact, "on_boundary")]


def component_bc(Z, uexact):
    return [DirichletBC(Z.sub(0).sub(0), uexact[0], [1, 2]),
            DirichletBC(Z.sub(0).sub(1), uexact[1], [3, 4])]


@pytest.mark.parametrize("butcher_tableau", [Alexander()] + wsodirks)
@pytest.mark.parametrize("bctype", [subspace_bc, component_bc])
def test_stokes_bcs(butcher_tableau, bctype):
    N = 10
    mesh = UnitSquareMesh(N, N)

    Ve = VectorElement("CG", mesh.ufl_cell(), 2)
    Pe = FiniteElement("CG", mesh.ufl_cell(), 1)
    Ze = MixedElement([Ve, Pe])
    Z = FunctionSpace(mesh, Ze)

    MC = MeshConstant(mesh)
    t = MC.Constant(0.0)
    dt = MC.Constant(1.0/N)
    (x, y) = SpatialCoordinate(mesh)

    uexact = as_vector([x*t + y**2, -y*t+t*(x**2)])
    pexact = x + y * t

    u_rhs = Dt(uexact) - div(grad(uexact)) + grad(pexact)
    p_rhs = -div(uexact)

    z = Function(Z)
    z_dirk = Function(Z)
    test_z = TestFunction(Z)
    (u, p) = split(z)
    (u_dirk, p_dirk) = split(z_dirk)
    (v, q) = split(test_z)
    F = (inner(Dt(u), v)*dx
         + inner(grad(u), grad(v))*dx
         - inner(p, div(v))*dx
         - inner(q, div(u))*dx
         - inner(u_rhs, v)*dx
         - inner(p_rhs, q)*dx)
    Fdirk = replace(F, {z: z_dirk})

    nsp = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=mesh.comm)])

    u, p = z.subfunctions
    u.interpolate(uexact)
    u_dirk, p_dirk = z_dirk.subfunctions
    u_dirk.interpolate(uexact)

    bcs = bctype(Z, uexact)

    lu = {"mat_type": "aij",
          "snes_type": "ksponly",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_shift_type": "inblocks"
          }

    stepper = TimeStepper(F, butcher_tableau, t, dt, z,
                          bcs=bcs, solver_parameters=lu, nullspace=nsp)

    stepperdirk = TimeStepper(
        Fdirk, butcher_tableau, t, dt, z_dirk,
        bcs=bcs, solver_parameters=lu, nullspace=nsp,
        stage_type="dirk")

    for i in range(10):
        stepper.advance()
        stepperdirk.advance()
        t.assign(float(t) + float(dt))
        assert errornorm(u_dirk, u) < 2.e-7
