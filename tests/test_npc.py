import pytest
from petsc4py import PETSc
import firedrake
from firedrake import (
    Constant, Function, FunctionSpace, SpatialCoordinate,
    TestFunction, UnitIntervalMesh, conditional, exp, cos,
    ds, dx, grad, inner
)
from irksome import BackwardEuler, Dt, IRKAuxiliaryOperatorSNES, TimeStepper, lag

# Stefan problem: a two-phase heat equation with a discontinuous conductivity
#
#   k(T) = k_solid  if T <= T_m  else  k_liquid.
#
# The fully implicit residual is not differentiable in T, so Newton's method
# fails. Lagging the conductivity to the start of the timestep gives a residual
# with a well-defined Jacobian. Here we use the lagged residual as a nonlinear
# preconditioner so that the outer solver converges to the fully implicit
# problem.

T_m = Constant(0.0)
k_solid = Constant(2.0)
k_liquid = Constant(1.0)
h = Constant(0.01)            # Robin penalty length scale
dT = Constant(0.25)           # amplitude of the boundary oscillation
T_scale = Constant(5e-4)      # scale for transition between solid and liquid


def stefan_conductivity(T):
    r = exp(-T / T_scale)
    s = 1 / (1 + r)
    return k_solid * (1 - s) + s * k_liquid


def stefan_form(T, q, t, k):
    """Robin residual with the oscillating boundary temperatures."""
    T_1 = Constant(-1.0)
    T_2 = Constant(1.0) + dT * cos(t)
    F_cells = (Dt(T) * q + k * inner(grad(T), grad(q))) * dx
    F_boundaries = k / h * (T - T_1) * q * ds(1) + k / h * (T - T_2) * q * ds(2)
    return F_cells + F_boundaries


class StefanLagAuxSNES(IRKAuxiliaryOperatorSNES):
    """Precondition the fully implicit Stefan problem with the residual whose
    conductivity is lagged to the start of the timestep."""
    def getNewForm(self, snes, T, q):
        t = self.get_appctx(snes)["stepper"].t
        k = lag(stefan_conductivity(T))
        return stefan_form(T, q, t, k), None


def stefan_setup():
    nx = 64
    mesh = UnitIntervalMesh(nx)
    Q = FunctionSpace(mesh, "CG", 1)
    x, = SpatialCoordinate(mesh)

    T = Function(Q)
    T.interpolate((1 - x) * Constant(-1.0) + x * Constant(1.0))

    t = Constant(0.0)
    dt = Constant(0.1)
    q = TestFunction(Q)
    k = stefan_conductivity(T)   # NOTE: this is the fully implicit form
    F = stefan_form(T, q, t, k)
    return F, t, dt, T


def test_npc_stefan_newton_fails():
    """Newton's method fails on the fully implicit Stefan problem because it
    isn't differentiable."""
    F, t, dt, T = stefan_setup()
    method = BackwardEuler()
    params = {
        "solver_parameters": {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_monitor": None,
        },
    }
    stepper = TimeStepper(F, method, t, dt, T, **params)
    with pytest.raises(firedrake.ConvergenceError):
        final_time = 20.0
        num_steps = int(final_time / float(dt))
        for step in range(num_steps):
            stepper.advance()
            t.assign(float(t) + float(dt))


def test_npc_stefan():
    """The lagged residual, used as a nonlinear preconditioner, drives the
    fully implicit Stefan problem to convergence where Newton fails."""
    F, t, dt, T = stefan_setup()
    params = {
        "solver_parameters": {
            "snes_type": "ngmres",
            "snes_monitor": None,
            "snes_max_it": 100,
            "npc": {
                "snes_type": "python",
                "snes_python_type": "test_npc.StefanLagAuxSNES",
                "aux": {
                    "snes_type": "ksponly",
                    "ksp_type": "cg",
                    "pc_type": "lu",
                },
            },
        },
    }
    method = BackwardEuler()
    stepper = TimeStepper(F, method, t, dt, T, **params)

    reasons = [
        PETSc.SNES.ConvergedReason.CONVERGED_FNORM_ABS,
        PETSc.SNES.ConvergedReason.CONVERGED_FNORM_RELATIVE,
    ]

    final_time = 20.0
    num_steps = int(final_time / float(dt))
    for step in range(num_steps):
        stepper.advance()
        t.assign(t + dt)
        assert stepper.solver.snes.getConvergedReason() in reasons
