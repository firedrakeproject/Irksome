import pytest
from firedrake import *
from irksome import Dt, LobattoIIIC, MeshConstant, RadauIIA, TimeStepper
from irksome.tools import AI, IA
from ufl.algorithms import expand_derivatives

# test the accuracy of the 2d Stokes heat equation using CG elements
# and LobattoIIIC time integration.  Particular issue being tested is
# enforcement of the pressure BC, which is a field that doesn't have
# a time derivative on it.


def StokesTest(N, butcher_tableau, stage_type="deriv", splitting=AI):
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

    u_rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) + grad(pexact)
    p_rhs = -div(uexact)

    z = Function(Z)
    test_z = TestFunction(Z)
    (u, p) = split(z)
    (v, q) = split(test_z)
    F = (inner(Dt(u), v)*dx
         + inner(grad(u), grad(v))*dx
         - inner(p, div(v))*dx
         - inner(q, div(u))*dx
         - inner(u_rhs, v)*dx
         - inner(p_rhs, q)*dx)

    bcs = [DirichletBC(Z.sub(0), uexact, "on_boundary")]
    nsp = [(1, VectorSpaceBasis(constant=True))]

    u, p = z.subfunctions
    u.interpolate(uexact)

    lu = {"mat_type": "aij",
          "snes_type": "ksponly",
          "snes_force_iteration": 1,
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps",
          "pc_factor_shift_type": "nonzero"}

    stepper = TimeStepper(F, butcher_tableau, t, dt, z,
                          stage_type=stage_type,
                          bcs=bcs, solver_parameters=lu)#, nullspace=nsp)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    (u, p) = z.subfunctions

    # check error mod the constants
    perr = norm(pexact - p - assemble((pexact-p) * dx))

    print(perr, errornorm(uexact, u))
    return errornorm(uexact, u) + perr


# make sure things run on a driven cavity.  We time step a while
# and check that the velocity is the right size (as observed from
# a "by-hand" backward Euler code in Firedrake
def NSETest(butch, stage_type, splitting):
    N = 16
    M = UnitSquareMesh(N, N)

    V = VectorFunctionSpace(M, "CG", 2)
    W = FunctionSpace(M, "CG", 1)
    Z = V * W

    up = Function(Z)
    u, p = split(up)
    v, q = TestFunctions(Z)

    Re = Constant(100.0)

    F = (inner(Dt(u), v) * dx
         + 1.0 / Re * inner(grad(u), grad(v)) * dx
         + inner(dot(grad(u), u), v) * dx
         - p * div(v) * dx
         + div(u) * q * dx
         )

    bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
           DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

    nullspace = [(1, VectorSpaceBasis(constant=True))]

    solver_parameters = {"mat_type": "aij",
                         "snes_type": "newtonls",
                         "snes_linesearch_type": "bt",
                         "snes_rtol": 1e-10,
                         "snes_atol": 1e-10,
                         "snes_force_iteration": 1,
                         "ksp_type": "preonly",
                         "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps"}

    MC = MeshConstant(M)
    t = MC.Constant(0.0)
    dt = MC.Constant(1.0/N)
    stepper = TimeStepper(F, butch,
                          t, dt, up,
                          bcs=bcs,
                          stage_type="value",
                          solver_parameters=solver_parameters,
                          nullspace=nullspace)

    tfinal = 1.0
    u, p = up.subfunctions
    while (float(t) < tfinal):
        if (float(t) + float(dt) > tfinal):
            dt.assign(1.0 - float(t))
        stepper.advance()
        p -= assemble(p*dx)
        print(float(t), norm(u), norm(p), assemble(p*dx))
        t.assign(float(t) + float(dt))

    return norm(u)


@pytest.mark.parametrize('stage_type', ("deriv", "value"))
@pytest.mark.parametrize('splitting', (AI, IA))
@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
@pytest.mark.parametrize('time_stages', (2, 3))
@pytest.mark.parametrize('butch', (LobattoIIIC, RadauIIA))
def test_Stokes(N, butch, time_stages, stage_type, splitting):
    error = StokesTest(N, butch(time_stages), stage_type, splitting)
    assert abs(error) < 4e-9


@pytest.mark.parametrize('stage_type', ("deriv", "value"))
@pytest.mark.parametrize('time_stages', (2,))
@pytest.mark.parametrize('butch', (LobattoIIIC, RadauIIA))
def test_NSE(butch, time_stages, stage_type):
    unrm = NSETest(butch(time_stages), stage_type, IA)
    assert abs(unrm - 0.216) < 5e-3


if __name__ == "__main__":
    test_Stokes(8, RadauIIA, 2, "value", IA)
