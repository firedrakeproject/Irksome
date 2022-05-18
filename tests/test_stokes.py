import pytest
from firedrake import *

from irksome import Dt, TimeStepper, LobattoIIIC
from irksome.tools import AI, IA
from ufl.algorithms import expand_derivatives

# test the accuracy of the 2d Stokes heat equation using CG elements
# and LobattoIIIC time integration.  Particular issue being tested is
# enforcement of the pressure BC, which is a field that doesn't have
# a time derivative on it.


def StokesTest(N, butcher_tableau, splitting=AI):
    mesh = UnitSquareMesh(N, N)

    Ve = VectorElement("CG", mesh.ufl_cell(), 2)
    Pe = FiniteElement("CG", mesh.ufl_cell(), 1)
    Ze = MixedElement([Ve, Pe])
    Z = FunctionSpace(mesh, Ze)

    t = Constant(0.0)
    dt = Constant(1.0/N)
    (x, y) = SpatialCoordinate(mesh)

    uexact = as_vector([x*t + y**2, -y*t+t*(x**2)])
    pexact = Constant(0, domain=mesh)

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

    u, p = z.split()
    u.interpolate(uexact)

    lu = {"mat_type": "aij",
          "snes_type": "newtonls",
          "snes_linesearch_type": "l2",
          "snes_linesearch_monitor": None,
          "snes_monitor": None,
          "snes_rtol": 1e-8,
          "snes_atol": 1e-8,
          "snes_force_iteration": 1,
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"}

    stepper = TimeStepper(F, butcher_tableau, t, dt, z,
                          bcs=bcs, solver_parameters=lu, nullspace=nsp)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    (u, p) = z.split()
    return errornorm(uexact, u) + errornorm(pexact, p)


@pytest.mark.parametrize('splitting', (AI, IA))
@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
@pytest.mark.parametrize('time_stages', (2, 3))
def test_Stokes(N, time_stages, splitting):
    error = StokesTest(N, LobattoIIIC(time_stages), splitting)
    assert abs(error) < 2e-10


if __name__ == "__main__":
    test_Stokes(4, 2, AI)
