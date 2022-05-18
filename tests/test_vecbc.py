import pytest
from firedrake import *
from irksome import GaussLegendre, Dt, TimeStepper
from irksome.tools import AI, IA


def elastodynamics(N, deg, butcher_tableau, splitting=AI):
    dt = Constant(1.0 / N)
    t = Constant(0.0)

    msh = UnitSquareMesh(N, N)
    x, y = SpatialCoordinate(msh)

    V = VectorFunctionSpace(msh, "CG", 1)
    VV = V * V

    uu1 = Function(VV)
    u1, udot1 = uu1.split()
    u1.interpolate(as_vector([sin(pi*x)*sin(pi*y), sin(pi*x) * sin(pi*y)]))

    uu2 = Function(VV)
    u2, udot2 = uu2.split()
    u2.interpolate(as_vector([sin(pi*x)*sin(pi*y), sin(pi*x) * sin(pi*y)]))

    u1, udot1 = split(uu1)
    u2, udot2 = split(uu2)

    v, w = TestFunctions(VV)

    F1 = (inner(Dt(u1) - udot1, v) * dx
          + inner(Dt(udot1), w) * dx + inner(grad(u1), grad(w)) * dx)

    F2 = (inner(Dt(u2) - udot2, v) * dx
          + inner(Dt(udot2), w) * dx + inner(grad(u2), grad(w)) * dx)

    bc1 = DirichletBC(VV.sub(0), Function(VV.sub(0)), "on_boundary")
    bc2 = [DirichletBC(VV.sub(0).sub(i), Function(VV.sub(0).sub(i)), "on_boundary")
           for i in (0, 1)]

    luparams = {"mat_type": "aij",
                "snes_type": "ksponly",
                "ksp_type": "preonly",
                "pc_type": "lu"}

    stepper1 = TimeStepper(F1, butcher_tableau, t, dt, uu1, bcs=bc1,
                           solver_parameters=luparams,
                           splitting=splitting)

    stepper2 = TimeStepper(F2, butcher_tableau, t, dt, uu2, bcs=bc2,
                           solver_parameters=luparams,
                           splitting=splitting)

    stepper1.advance()
    stepper2.advance()
    return norm(uu1-uu2)


@pytest.mark.parametrize('splitting', (AI, IA))
@pytest.mark.parametrize('time_stages', (1, 2))
def test_bc(time_stages, splitting):
    assert elastodynamics(4, 1, GaussLegendre(time_stages), splitting)
