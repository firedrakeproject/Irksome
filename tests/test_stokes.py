import pytest
import numpy as np
from firedrake import *

from irksome import Dt, TimeStepper, LobattoIIIC
from ufl.algorithms import expand_derivatives

# test the accuracy of the 1d heat equation using CG elements
# and Gauss-Legendre time integration


def StokesTest(N, butcher_tableau):
    mesh = UnitSquareMesh(N,N)

    Ve = VectorElement("CG", mesh.ufl_cell(), 2)
    Pe = FiniteElement("CG", mesh.ufl_cell(), 1)
    Ze = MixedElement([Ve, Pe])
    Z = FunctionSpace(mesh, Ze)

    t = Constant(0.0)
    dt = Constant(1.0/N)
    (x,y) = SpatialCoordinate(mesh)

    uexact = as_vector([x*t + y**2,-y*t+t*(x**2)])
    pexact = Constant(0, domain=mesh)

    u_rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) + grad(pexact)
    p_rhs = -div(uexact)

    z = Function(Z)
    test_z = TestFunction(Z)
    (u, p) = split(z)
    (v, q) = split(test_z)
    F = (  inner(Dt(u),v)*dx
         + inner(grad(u), grad(v))*dx
         - inner(p, div(v))*dx
         - inner(q, div(u))*dx
         - inner(u_rhs, v)*dx
         - inner(p_rhs, q)*dx
        )
    
    bcs = [
           DirichletBC(Z.sub(0), uexact, "on_boundary"),
           DirichletBC(Z.sub(1), pexact, "on_boundary")
          ]

    u,p = z.split()
    u.interpolate(uexact)

    lu = {
       "mat_type": "aij",
       "snes_type": "newtonls",
       "snes_linesearch_type": "l2",
       "snes_linesearch_monitor": None,
       "snes_monitor": None,
       "snes_rtol": 1e-8,
       "snes_atol": 1e-8,
       "ksp_type": "preonly",
       "pc_type": "lu",
       "pc_factor_mat_solver_type": "mumps",
         }

    stepper = TimeStepper(F, butcher_tableau, t, dt, z,
                          bcs=bcs, solver_parameters=lu)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    (u,p) = z.split()
    return errornorm(uexact, u)

@pytest.mark.parametrize(('N', 'time_stages'),
                         [(2**j, i) for j in range(2, 4)
                          for i in (2, 3)])
def test_Stokes(N, time_stages):
    error = StokesTest(N, LobattoIIIC(time_stages))
    assert abs(error) < 1e-10


if __name__ == "__main__":
    test_Stokes(4, 2)
