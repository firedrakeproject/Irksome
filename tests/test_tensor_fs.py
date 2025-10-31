from irksome import GaussLegendre, Dt, TimeStepper
from firedrake import *
import pytest


@pytest.mark.parametrize("stage_type", ["deriv", "value"])
def test_tensor(stage_type):
    butcher_tableau = GaussLegendre(1)

    msh = UnitSquareMesh(1, 1)
    V = TensorFunctionSpace(msh, "CG", 1)

    dt = Constant(1.0)
    t = Constant(0.0)

    u = Function(V)
    v = TestFunction(V)

    F = inner(Dt(u), v)*dx + inner(u, v)*dx
    luparams = {"mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu"}
    bc = []
    stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                          stage_type=stage_type,
                          solver_parameters=luparams)
    stepper.advance()
