import numpy as np
import pytest
from firedrake import *
from irksome import WSODIRK, Alexander, Dt, LobattoIIIA, MeshConstant, TimeStepper
from irksome.tableaux.ButcherTableaux import ButcherTableau
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

    bnd_error = inner(u-uexact, u-uexact) * ds
    t_end = 2.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        # Check solution and boundary values
        assert errornorm(uexact, u) / norm(uexact) < 1e-3
        assert abs(assemble(bnd_error)) ** 0.5 < 1e-12


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
    (v, q) = split(test_z)
    F = (inner(Dt(u), v)*dx
         + inner(grad(u), grad(v))*dx
         - inner(p, div(v))*dx
         - inner(q, div(u))*dx
         - inner(u_rhs, v)*dx
         - inner(p_rhs, q)*dx)

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

    # Test LinearVariationalSolver interface
    trial_z = TrialFunction(Z)
    Fdirk = replace(F, {z: trial_z})
    stepperdirk = TimeStepper(
        Fdirk, butcher_tableau, t, dt, z_dirk,
        bcs=bcs, solver_parameters=lu, nullspace=nsp,
        stage_type="dirk")

    for i in range(10):
        stepper.advance()
        stepperdirk.advance()
        t.assign(float(t) + float(dt))
        assert errornorm(u_dirk, u) < 2.e-7


def dirk_heat_problem():
    msh = UnitIntervalMesh(4)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(0.05)
    (x,) = SpatialCoordinate(msh)
    u = Function(V).interpolate(x * (1 - x))
    v = TestFunction(V)
    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
    return F, t, dt, u


@pytest.mark.parametrize("kwarg", ["J", "Jp"])
def test_dirk_accepts_jacobian_forms(kwarg):
    F, t, dt, u = dirk_heat_problem()
    stepper = TimeStepper(F, Alexander(), t, dt, u, stage_type="dirk",
                          **{kwarg: derivative(F, u)})
    assert len(stepper.problem.J.arguments()) == 2


def test_dirk_rejects_constant_jacobian():
    F, t, dt, u = dirk_heat_problem()
    with pytest.raises(ValueError):
        TimeStepper(F, Alexander(), t, dt, u, stage_type="dirk",
                    constant_jacobian=True)


def tr_bdf2():
    gamma = 2 - np.sqrt(2)
    w = np.sqrt(2) / 4
    A = np.array([[0, 0, 0], [gamma / 2, gamma / 2, 0], [w, w, gamma / 2]])
    return ButcherTableau(A, A[-1], None, np.array([0, gamma, 1]), 2, None, None)


@pytest.mark.parametrize("butcher_tableau", [LobattoIIIA(2), tr_bdf2()], ids=["LobattoIIIA(2)", "TR-BDF2"])
def test_explicit_first_stage_dirichletbc(butcher_tableau):
    """A zero diagonal entry leaves its stage value unconstrained.

    Stage 0 of these methods has A[0, 0] == 0, so it cannot move its own
    stage value and its stage derivative is set to zero on the boundary.
    The methods are stiffly accurate, so the last stage imposes the data.
    """
    msh = UnitIntervalMesh(8)
    V = FunctionSpace(msh, "CG", 2)
    (x,) = SpatialCoordinate(msh)
    t_end = 1.0

    def error(nsteps):
        MC = MeshConstant(msh)
        t = MC.Constant(0.0)
        dt = MC.Constant(t_end / nsteps)
        uexact = 2.0 + x * (1.0 + atan(t)) + cos(t) * x**2
        rhs = Dt(uexact) - div(grad(uexact))
        u = Function(V).interpolate(uexact)
        v = TestFunction(V)
        F = (inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
             - inner(rhs, v) * dx)
        stepper = TimeStepper(F, butcher_tableau, t, dt, u,
                              bcs=DirichletBC(V, uexact, "on_boundary"),
                              stage_type="dirk")
        for _ in range(nsteps):
            stepper.advance()
            t.assign(float(t) + float(dt))
        return errornorm(uexact, u)

    errors = np.array([error(10 * 2**r) for r in range(3)])
    rates = np.diff(-np.log2(errors))
    assert (rates > butcher_tableau.order - 0.25).all()


@pytest.mark.parametrize("A,b,stage_type", [
    ([[0, 0], [0.5, 0.5]], [0.25, 0.75], "dirk"),
    ([[1/3, 0, 0], [2/3, 0, 0], [1, -0.5, 0.5]], [1, -0.5, 0.5], "dirk"),
    ([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], [0, 0.5, 0.5], "explicit")],
    ids=["not stiffly accurate", "zero diagonal after the first stage", "zero subdiagonal"])
def test_zero_diagonal_boundary_data(A, b, stage_type):
    A = np.array(A)
    butcher_tableau = ButcherTableau(A, np.array(b), None, A.sum(axis=1), 1, None, None)
    msh = UnitIntervalMesh(4)
    V = FunctionSpace(msh, "CG", 1)
    t = Constant(0.0)
    dt = Constant(0.1)
    u = Function(V)
    v = TestFunction(V)
    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
    stepper = TimeStepper(F, butcher_tableau, t, dt, u, stage_type=stage_type,
                          bcs=DirichletBC(V, 0, "on_boundary"))
    stepper.advance()
    with pytest.raises(NotImplementedError):
        TimeStepper(F, butcher_tableau, t, dt, u, stage_type=stage_type,
                    bcs=DirichletBC(V, 1 + t, "on_boundary"))
