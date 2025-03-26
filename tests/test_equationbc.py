import pytest
from firedrake import *
from math import isclose
from irksome import GaussLegendre, RadauIIA, Dt, MeshConstant, TimeStepper
from ufl.algorithms.ad import expand_derivatives


@pytest.mark.parametrize("stage_type", ["deriv"])
@pytest.mark.parametrize("butcher_tableau", [GaussLegendre(3)])
def test_1d_heat_dirichletbc(butcher_tableau, stage_type):

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
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))
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
        F, butcher_tableau, t, dt, u, bcs=bc, solver_parameters=luparams, bc_type="DAE"
    )

    t_end = 2.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        # Check solution and boundary values
        assert norm(u - uexact) / norm(uexact) < 1e-5
        assert isclose(u.at(x0), u_0)
        assert isclose(u.at(x1), u_1)


@pytest.mark.parametrize("stage_type", ["deriv"])
@pytest.mark.parametrize("butcher_tableau", [RadauIIA(2)])
def test_2d_heat_mixed_robinbc_nonlinear(butcher_tableau, stage_type):

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
    F = (inner(Dt(u), v)*dx + inner(div(sigma), v)*dx - inner(rhs, v)*dx(degree=10)
        + inner(sigma, tau)*dx - inner(u, div(tau))*dx)

    bc = EquationBC(inner(u + (u ** 2) - dot(sigma, nml) - bdrydata, dot(tau, nml)) * ds == 0, sln, (1, 2, 3, 4), V=Z.sub(1))

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = TimeStepper(
        F, butcher_tableau, t, dt, sln, bcs=bc, solver_parameters=luparams, bc_type="DAE"
    )

    u, sigma = sln.subfunctions
    u.interpolate(uexact)

    t_end = 1.0
    while float(t) < t_end:
        stepper.advance()
        t.assign(float(t) + float(dt))
        # Check solution values
        assert errornorm(uexact, u) / norm(uexact) < 1e-3
        assert errornorm(sigmaexact, sigma) / norm(sigmaexact) < 1e-3
