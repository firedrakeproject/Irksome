import pytest
from firedrake import *
from irksome import (ContinuousPetrovGalerkinScheme, Dt,
                     GalerkinCollocationScheme, GaussLegendre,
                     MeshConstant, RadauIIA, TimeStepper)

# TODO: Should also test with aux_indices

def galerkin_methods():
    return [
        pytest.param(
            ContinuousPetrovGalerkinScheme(
                1, quadrature_degree=12, basis_type="integral"),
            id="cpg1-integral"),
        pytest.param(
            ContinuousPetrovGalerkinScheme(
                2, quadrature_degree=12, basis_type="integral"),
            id="cpg2-integral"),
        pytest.param(
            ContinuousPetrovGalerkinScheme(
                2, quadrature_degree=12, basis_type="Lagrange"),
            id="cpg2-lagrange"),
        pytest.param(
            ContinuousPetrovGalerkinScheme(
                2, quadrature_degree=12, basis_type="Bernstein"),
            id="cpg2-bernstein"),
        pytest.param(
            ContinuousPetrovGalerkinScheme(
                3, quadrature_degree=12, basis_type="integral"),
            id="cpg3-integral"),
        pytest.param(
            GalerkinCollocationScheme(
                1, stage_type="deriv", quadrature_degree=12),
            id="collocation1-deriv"),
        pytest.param(
            GalerkinCollocationScheme(
                2, stage_type="value", quadrature_degree=12),
            id="collocation2-value"),
        pytest.param(
            GalerkinCollocationScheme(
                2, stage_type="deriv", quadrature_degree=12,
                quadrature_scheme="radau"),
            id="collocation2-radau-deriv"),
        pytest.param(
            GalerkinCollocationScheme(
                3, stage_type="value", quadrature_degree=12,
                quadrature_scheme="lobatto"),
            id="collocation3-lobatto-value"),
    ]


def mixed_galerkin_methods():
    return [
        pytest.param(
            ContinuousPetrovGalerkinScheme(
                2, quadrature_degree=12, basis_type="Lagrange"),
            id="cpg2-lagrange"),
        pytest.param(
            ContinuousPetrovGalerkinScheme(
                2, quadrature_degree=12, basis_type="Bernstein"),
            id="cpg2-bernstein"),
        pytest.param(
            ContinuousPetrovGalerkinScheme(
                3, quadrature_degree=12, basis_type="Lagrange"),
            id="cpg3-lagrange"),
        pytest.param(
            GalerkinCollocationScheme(
                2, stage_type="value", quadrature_degree=12),
            id="collocation2-value"),
        pytest.param(
            GalerkinCollocationScheme(
                2, stage_type="deriv", quadrature_degree=12,
                quadrature_scheme="radau"),
            id="collocation2-radau-deriv"),
        pytest.param(
            GalerkinCollocationScheme(
                3, stage_type="value", quadrature_degree=12,
                quadrature_scheme="lobatto"),
            id="collocation3-lobatto-value"),
    ]


def nested_bc_galerkin_methods():
    return [
        pytest.param(
            ContinuousPetrovGalerkinScheme(1, basis_type="integral"),
            id="cpg1-integral"),
        pytest.param(
            ContinuousPetrovGalerkinScheme(2, basis_type="Lagrange"),
            id="cpg2-lagrange"),
        pytest.param(
            ContinuousPetrovGalerkinScheme(3, basis_type="Lagrange"),
            id="cpg3-lagrange"),
        pytest.param(
            GalerkinCollocationScheme(1, stage_type="deriv"),
            id="collocation1-deriv"),
        pytest.param(
            GalerkinCollocationScheme(2, stage_type="value"),
            id="collocation2-value"),
        pytest.param(
            GalerkinCollocationScheme(
                2, stage_type="deriv", quadrature_scheme="radau"),
            id="collocation2-radau-deriv"),
        pytest.param(
            GalerkinCollocationScheme(
                3, stage_type="value", quadrature_degree=12,
                quadrature_scheme="lobatto"),
            id="collocation3-lobatto-value"),
    ]


@pytest.mark.parametrize(
    "method",
    [pytest.param(GaussLegendre(3), id="gauss-legendre3")]
    + galerkin_methods())
def test_1d_heat_equationbc(method):

    # Boundary values
    u_0 = Constant(2.0)
    u_1 = Constant(3.0)

    N = 100
    x0 = 0.0
    x1 = 10.0
    msh = IntervalMesh(N, x1)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    dt = MC.Constant(10.0 / N)
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
        EquationBC(inner(u - u_1, v) * ds(2) == 0, u, 2),
        EquationBC(inner(u - u_0, v) * ds(1) == 0, u, 1),
    ]

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = TimeStepper(
        F, method, t, dt, u, bcs=bc,
        solver_parameters=luparams, bc_type="DAE")

    bnd_error = inner(u-uexact, u-uexact) * ds
    t_end = 2.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        # Check solution and boundary values
        assert norm(u - uexact) / norm(uexact) < 1e-5
        assert abs(assemble(bnd_error)) ** 0.5 < 1e-12


@pytest.mark.parametrize(
    "method",
    [pytest.param(RadauIIA(2), id="radau-iia2")]
    + mixed_galerkin_methods())
def test_2d_heat_mixed_robinbc_nonlinear(method):

    N = 10

    msh = UnitSquareMesh(N, N)
    nml = FacetNormal(msh)

    Q = FunctionSpace(msh, "DG", 2)
    V = FunctionSpace(msh, "BDM", 3)

    Z = Q * V

    x, y = SpatialCoordinate(msh)

    MC = MeshConstant(msh)
    dt = MC.Constant(1 / N)
    t = MC.Constant(0.0)

    uexact = sin(t) * cos(x + y + t)
    sigmaexact = -grad(uexact)

    rhs = Dt(uexact) + div(sigmaexact)
    bdrydata = uexact + (uexact ** 2) - dot(sigmaexact, nml)

    sln = Function(Z)
    u, sigma = split(sln)

    v, tau = TestFunctions(Z)
    F = (inner(Dt(u) + div(sigma), v)*dx - inner(rhs, v)*dx(degree=10)
         + inner(sigma, tau)*dx - inner(u, div(tau))*dx)

    bc = EquationBC(inner(u + (u ** 2) - dot(sigma, nml) - bdrydata, dot(tau, nml)) * ds == 0, sln, (1, 2, 3, 4), V=Z.sub(1))

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = TimeStepper(
        F, method, t, dt, sln, bcs=bc,
        solver_parameters=luparams, bc_type="DAE"
    )

    u, sigma = sln.subfunctions
    u.interpolate(uexact)
    sigma.interpolate(sigmaexact)

    t_end = 1.0
    while float(t) < t_end:
        stepper.advance()
        t.assign(float(t) + float(dt))
        # Check solution values
        assert errornorm(uexact, u) / norm(uexact) < 1e-3
        assert errornorm(sigmaexact, sigma) / norm(sigmaexact) < 1e-3


@pytest.mark.parametrize(
    "method",
    [pytest.param(RadauIIA(2), id="radau-iia2"),
     pytest.param(GaussLegendre(3), id="gauss-legendre3")]
    + nested_bc_galerkin_methods())
def test_2d_equation_and_dirichlet_bc(method):

    msh = UnitSquareMesh(5, 5)
    V = FunctionSpace(msh, "CG", 4)

    x, y = SpatialCoordinate(msh)

    MC = MeshConstant(msh)
    dt = MC.Constant(0.01)
    t = MC.Constant(0.0)

    uexact = cos(t) * cos(x + y + t)
    rhs = Dt(uexact) - div(grad(uexact))

    u = Function(V)
    v = TestFunction(V)

    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx(degree=10)

    bc0 = DirichletBC(V, uexact, ((1, 3), (1, 4), (2, 3), (2, 4)))
    bc1 = EquationBC(inner((u ** 2) - (uexact ** 2), v) * ds((1, 2), degree=10) == 0, u, (1, 2), bcs=[bc0])
    bc2 = DirichletBC(V, uexact, (3, 4))

    params = {"snes_rtol": 1e-12, "snes_atol": 1e-12, "ksp_type": "preonly", "pc_type": "lu"}

    # Galerkin methods solve for stage values, so initialise the solution
    # before constructing the stepper and its stage initial guess.
    u.interpolate(uexact)
    stepper = TimeStepper(
        F, method, t, dt, u, bcs=[bc1, bc2],
        solver_parameters=params, bc_type="DAE"
    )

    # Do a single timestep
    stepper.advance()
    t.assign(float(t) + float(dt))

    # Check that the BCs from bc0 are satisfied
    if isinstance(method, ContinuousPetrovGalerkinScheme):
        point_tol = 1e-4
        diff = Function(V)
        diff.interpolate(u - uexact)
        assert abs(diff.at([0, 0])) < point_tol
        assert abs(diff.at([1, 0])) < point_tol
        assert abs(diff.at([0, 1])) < point_tol
        assert abs(diff.at([1, 1])) < point_tol
        boundary_tol = 1e-4
    else:
        diff = Function(V)
        diff.interpolate(u - uexact)
        assert diff.at([0, 0]) < 1e-10
        assert diff.at([1, 0]) < 1e-10
        assert diff.at([0, 1]) < 1e-10
        assert diff.at([1, 1]) < 1e-10
        boundary_tol = 1e-6

    # Check that the BCs from bc1, bc2 are satisfied
    assert sqrt(assemble(
        ((u - uexact) ** 2) * ds((1, 2, 3, 4), degree=10))) < boundary_tol
