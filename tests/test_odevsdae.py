import pytest
from firedrake import (DirichletBC, Function, FunctionSpace,
                       TestFunction, UnitIntervalMesh,
                       inner, dx, grad, norm)
from irksome import Dt, TimeStepper, RadauIIA, LobattoIIIC


def run_heat(N, deg, bt, bc_type):
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", deg)
    t = Constant(0.0)
    dt = Constant(1 / N)

    u = Function(V)
    v = TestFunction(V)

    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx

    bc = DirichletBC(V, 1, "on_boundary")

    stepper = TimeStepper(F, bt, t, dt, u,
                          bcs=bc, bc_type=bc_type)

    while (float(t) < 1.0):
        stepper.advance()
        t.assign(float(t) + float(dt))

    return norm(u)


# Test a "feature" (really, bug) that ODE
# BC can be off by a constant shift
# solution should stay fixed at 0.
@pytest.mark.parametrize('N', (8,))
@pytest.mark.parametrize('deg', (1, 2))
@pytest.mark.parametrize('bt', (RadauIIA(1),
                                RadauIIA(2),
                                LobattoIIIC(2),
                                LobattoIIIC(3)))
def test_ode_bc(N, deg, bt):
    assert run_heat(N, deg, bt, "ODE") < 1.e-10


# Test that DAE BC handle incompatible BC + IC
# just fine (unlike ODE).  Solution should be reasonably
# close to u=1 after one unit of time.
@pytest.mark.parametrize('N', (8,))
@pytest.mark.parametrize('deg', (1, 2))
@pytest.mark.parametrize('bt', (RadauIIA(1),
                                RadauIIA(2),
                                LobattoIIIC(2),
                                LobattoIIIC(3)))
def test_dae_bc(N, deg, bt):
    assert abs(run_heat(N, deg, bt, "DAE")-1.0) < 1.e-2
