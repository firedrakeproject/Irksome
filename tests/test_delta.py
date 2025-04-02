from firedrake import *
from irksome import Alexander, Dt, RadauIIA, TimeStepper
import pytest
import ufl


def heat_delta(bt, stage_type):
    N = 4
    msh = UnitSquareMesh(N, N)
    V = FunctionSpace(msh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    t = Constant(0)

    vom_msh = VertexOnlyMesh(msh, [(0.5, 0.5)])
    Vvom = FunctionSpace(vom_msh, "DG", 0)
    vvom = TestFunction(Vvom)

    Fvom = inner(-cos(t), vvom) * dx

    d = Interpolate(v, Fvom)
    assert d.arguments()[0] == v

    Vbig = V * V
    test = TestFunction(Vbig)
    d0 = ufl.replace(d, {v: test[i]})
    assert d != d0, str(d)
    # print("d", d)
    # print("d0", d0)

    # c = assemble(d0)
    # assert c.function_space().dual() == Vbig
    # print(c.dat.data)

    F0 = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
    F = F0 + d

    Foo = inner(grad(u), grad(v)) * dx + d
    assemble(Foo)
    Joo = derivative(d, u)
    assemble(Joo)

    assert F.arguments() == (v,)

    assert len(d.arguments()) == 1
    assert len(F0.arguments()) == 1
    assert len(F.arguments()) == 1

    dt = Constant(1/N)

    stepper = TimeStepper(F, bt, t, dt, u, stage_type=stage_type)

    stepper.advance()


@pytest.mark.parametrize('stage_type', ('deriv', 'value'))
@pytest.mark.parametrize('num_stages', (1, 2))
def test_fi(num_stages, stage_type):
    heat_delta(RadauIIA(num_stages), stage_type)


def test_dirk():
    heat_delta(Alexander(), "dirk")
