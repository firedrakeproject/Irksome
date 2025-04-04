from firedrake import *
from irksome import Alexander, Dt, RadauIIA, TimeStepper, GaussLegendre, StageDerivativeNystromTimeStepper, GalerkinTimeStepper, DiscontinuousGalerkinTimeStepper
import pytest
import ufl


def delta(v, expr):
    msh = v.function_space().mesh()

    vom = VertexOnlyMesh(msh, [(0.5, 0.5)])
    P0 = FunctionSpace(vom, "DG", 0)
    v0 = TestFunction(P0)

    F = inner(expr, v0) * dx
    return Interpolate(v, F)


def test_delta():
    N = 4
    msh = UnitSquareMesh(N, N)
    V = FunctionSpace(msh, "CG", 1)
    v = TestFunction(V)
    u = Function(V)

    t = Constant(0)

    d = delta(v, cos(t))
    assert d.arguments() == (v,)

    # Test domain
    assert ufl.domain.extract_domains(d) == (msh,)

    # Test steady RHS and Jacobian assembly
    F = inner(grad(u), grad(v)) * dx - d
    assemble(F)
    J = derivative(F, u)
    assemble(J)

    # Test stage replacement
    Vbig = V * V
    test = TestFunction(Vbig)

    d0 = ufl.replace(d, {v: test[0]})
    assert d != d0
    assert d0.arguments() == (test,)

    # Test replacement assembly
    c = assemble(d0)
    assert c.function_space().dual() == Vbig

    R = FunctionSpace(msh, "R", 0)
    q = Function(R)  # Passes
    q = Constant(0)  # Fails
    d01 = ufl.replace(d, {v: q*test[1]})
    assert d != d01
    assert d01.arguments() == (test,)

    print(d01)
    # Test replacement assembly
    c = assemble(d01)
    assert c.function_space().dual() == Vbig


def heat_delta(bt, stage_type):
    N = 4
    msh = UnitSquareMesh(N, N)
    V = FunctionSpace(msh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    t = Constant(0)

    d = delta(v, cos(t))
    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - d
    assert F.arguments() == (v,)

    dt = Constant(1/N)

    stepper = TimeStepper(F, bt, t, dt, u, stage_type=stage_type)

    stepper.advance()


def heat_delta_galerkin(stepper, order):
    N = 4
    msh = UnitSquareMesh(N, N)
    V = FunctionSpace(msh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    t = Constant(0)
    d = delta(v, cos(t))

    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - d

    dt = Constant(1/N)

    stepper = stepper(F, order, t, dt, u)

    stepper.advance()


def wave_delta(bt):
    N = 4
    msh = UnitSquareMesh(N, N)
    V = FunctionSpace(msh, "CG", 1)
    u = Function(V)
    ut = Function(V)
    v = TestFunction(V)

    t = Constant(0)
    d = delta(v, cos(t))

    F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx - d

    dt = Constant(1/N)

    stepper = StageDerivativeNystromTimeStepper(F, bt, t, dt, u, ut)

    stepper.advance()


@pytest.mark.parametrize('stage_type', ('deriv', 'value'))
@pytest.mark.parametrize('num_stages', (1, 2))
def test_heat_fi(num_stages, stage_type):
    heat_delta(RadauIIA(num_stages), stage_type)


def test_heat_dirk():
    heat_delta(Alexander(), "dirk")


@pytest.mark.parametrize('num_stages', (1, 2))
def test_wave(num_stages):
    wave_delta(GaussLegendre(num_stages))


@pytest.mark.parametrize('stepper', (DiscontinuousGalerkinTimeStepper,
                                     GalerkinTimeStepper))
def test_galerkin(stepper):
    heat_delta_galerkin(stepper, 2)
