from firedrake import *
from firedrake.__future__ import interpolate
from irksome import Alexander, Dt, RadauIIA, TimeStepper
import pytest


def heat_delta(bt, stage_type):
    N = 4
    msh = UnitSquareMesh(N, N)
    V = FunctionSpace(msh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    t = Constant(0)

    vom_msh = VertexOnlyMesh(msh, [(0.5, 0.5)])
    Vvom = FunctionSpace(vom_msh, "DG", 0)
    delta = Function(Vvom).interpolate(1.0)

    vvom = TestFunction(Vvom)

    d = action(inner(cos(t) * delta, vvom) * dx, interpolate(v, Vvom))

    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - d

    dt = Constant(1/N)

    stepper = TimeStepper(F, bt, t, dt, u, stage_type=stage_type)

    stepper.advance()

@pytest.mark.parametrize('stage_type', ('deriv', 'value'))
@pytest.mark.parametrize('num_stages', (1, 2))
def test_fi(num_stages, stage_type):
    heat_delta(RadauIIA(num_stages), stage_type)


def test_dirk():
    heat_delta(Alexander(), "dirk")
