from firedrake import *
from irksome import Alexander, Dt, RadauIIA, TimeStepper, GaussLegendre, StageDerivativeNystromTimeStepper, GalerkinTimeStepper, DiscontinuousGalerkinTimeStepper
import pytest
import ufl


N = 4


@pytest.fixture
def msh():
    return UnitSquareMesh(N, N)


@pytest.fixture
def vom(msh):
    return VertexOnlyMesh(msh, [(0.5, 0.5)])


def delta(v, expr, vom):
    P0 = FunctionSpace(vom, "DG", 0)
    v_vom = TestFunction(P0)

    F_vom = inner(expr, v_vom) * dx
    return Interpolate(v, F_vom)


def test_delta(msh, vom):
    V = FunctionSpace(msh, "CG", 1)
    v = TestFunction(V)
    u = Function(V, name="u")

    t = Constant(0.3, name="time")

    d = delta(v, cos(t), vom)
    assert d.arguments() == (v,)

    # Test domain
    assert ufl.domain.extract_domains(d) == (msh,)

    # Test assembly
    result = assemble(d)
    x = msh.coordinates
    expected = np.logical_and(x.dat.data_ro[:, 0] == 0.5, x.dat.data_ro[:, 1] == 0.5).astype(float)
    expected *= np.cos(float(t))
    assert np.allclose(result.dat.data_ro, expected)

    # Test steady RHS and Jacobian assembly
    F = inner(grad(u), grad(v)) * dx - d
    assemble(F)
    J = derivative(F, u)
    assemble(J)

    # Test replacement with Constant and Function(R)
    c0 = Constant(1)
    c1 = Function(FunctionSpace(msh, "R", 0))
    dc = ufl.replace(d, {v: v * (c0 + c1)})
    assert d != dc
    assert dc.arguments() == (v,)

    # Test replacement assembly
    resultc = assemble(dc)
    assert resultc.function_space() == V.dual()
    assert np.allclose(result.dat.data_ro, float(c0 + c1)*expected)

    # Test stage replacement
    Vbig = V * V
    test = TestFunction(Vbig)
    d1 = ufl.replace(d, {v: test[1]})
    assert d != d1
    assert d1.arguments() == (test,)

    # Test replacement assembly
    result1 = assemble(d1)
    assert np.allclose(result1.dat[1].data_ro, result.dat.data_ro)
    assert result1.function_space() == Vbig.dual()


def heat_delta(msh, vom, bt, stage_type):
    V = FunctionSpace(msh, "CG", 1)
    u = Function(V, name="u")
    v = TestFunction(V)

    t = Constant(0)

    d = delta(v, cos(t*pi), vom)

    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - d
    bcs = DirichletBC(V, 0, "on_boundary")

    dt = Constant(1/N)

    stepper = TimeStepper(F, bt, t, dt, u, bcs=bcs, stage_type=stage_type, solver_parameters={"snes_lag_jacobian": -2})

    stepper.advance()


def heat_delta_galerkin(msh, vom, stepper, order):
    V = FunctionSpace(msh, "CG", 1)
    u = Function(V, name="u")
    v = TestFunction(V)

    t = Constant(0)
    d = delta(v, cos(t), vom)

    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - d
    bcs = DirichletBC(V, 0, "on_boundary")

    dt = Constant(1/N)

    stepper = stepper(F, order, t, dt, u, bcs=bcs, solver_parameters={"snes_lag_jacobian": -2})

    stepper.advance()


def wave_delta(msh, vom, bt):
    V = FunctionSpace(msh, "CG", 1)
    u = Function(V, name="u")
    ut = Function(V)
    v = TestFunction(V)

    t = Constant(0)
    d = delta(v, sin(t*pi)*Constant(0.1), vom)

    F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx - d
    bcs = DirichletBC(V, 0, "on_boundary")

    dt = Constant(1/N)

    stepper = StageDerivativeNystromTimeStepper(F, bt, t, dt, u, ut, bcs=bcs, solver_parameters={"snes_lag_jacobian": -2})

    stepper.advance()


@pytest.mark.parametrize('stage_type', ('deriv', 'value'))
@pytest.mark.parametrize('num_stages', (1, 2))
def test_heat_fully_implicit(msh, vom, num_stages, stage_type):
    heat_delta(msh, vom, RadauIIA(num_stages), stage_type)


def test_heat_dirk(msh, vom):
    heat_delta(msh, vom, Alexander(), "dirk")


@pytest.mark.parametrize('num_stages', (1, 2))
def test_wave(msh, vom, num_stages):
    wave_delta(msh, vom, GaussLegendre(num_stages))


@pytest.mark.parametrize('stepper,degree', [(DiscontinuousGalerkinTimeStepper, 1),
                                            (GalerkinTimeStepper, 2)])
def test_heat_galerkin(msh, vom, stepper, degree):
    heat_delta_galerkin(msh, vom, stepper, degree)
