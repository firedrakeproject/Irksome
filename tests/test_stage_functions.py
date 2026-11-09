"""stage_functions substitutes a coefficient stage by stage, on du/dt = w."""
import numpy as np
import pytest
from firedrake import (Constant, Function, FunctionSpace, TestFunction,
                       UnitIntervalMesh, assemble, dx, inner)
from irksome import Dt, RadauIIA, TimeStepper

stage_types = ("deriv", "value")
tableau = RadauIIA(2)
dt_value = 0.25


def source_problem():
    msh = UnitIntervalMesh(2)
    V = FunctionSpace(msh, "DG", 0)
    t = Constant(0.0)
    dt = Constant(dt_value)
    u = Function(V)
    w = Function(V)
    v = TestFunction(V)
    F = inner(Dt(u), v) * dx - inner(w, v) * dx
    return F, t, dt, u, w


def residual(stepper):
    Fbig, _ = stepper.get_form_and_bcs(stepper.stages)
    r = assemble(Fbig)
    return np.array([np.asarray(ri.dat.data_ro).ravel()
                     for ri in r.subfunctions])


def fill(W, values):
    for Wi, val in zip(W.subfunctions, values):
        Wi.assign(Constant(val))


@pytest.mark.parametrize("stage_type", stage_types)
def test_per_stage(stage_type):
    """Each stage of w moves the residual, and differently from the others."""
    F, t, dt, u, w = source_problem()
    stepper = TimeStepper(F, tableau, t, dt, u,
                          stage_type=stage_type, stage_functions=[w])

    W = stepper.stage_functions[w]
    num_stages = tableau.num_stages

    fill(W, [0.0] * num_stages)
    base = residual(stepper)

    columns = []
    for j in range(num_stages):
        fill(W, [1.0 if i == j else 0.0 for i in range(num_stages)])
        columns.append(residual(stepper) - base)

    for col in columns:
        assert np.linalg.norm(col) > 1e-12

    for j in range(num_stages):
        for k in range(j + 1, num_stages):
            assert np.linalg.norm(columns[j] - columns[k]) > 1e-12


@pytest.mark.parametrize("stage_type", stage_types)
def test_source(stage_type):
    """w(t) = t integrates du/dt = t exactly.

    Stage values take w(c_i*dt) = c_i*dt; stage derivatives take dw/dt = 1.
    """
    F, t, dt, u, w = source_problem()
    stepper = TimeStepper(F, tableau, t, dt, u,
                          stage_type=stage_type, stage_functions=[w])

    W = stepper.stage_functions[w]
    if stage_type == "value":
        fill(W, [float(ci) * dt_value for ci in tableau.c])
    else:
        fill(W, [1.0] * tableau.num_stages)

    stepper.advance()

    assert abs(assemble(u * dx) - dt_value ** 2 / 2) < 1e-12


def test_old_value():
    """to_value prepends w itself, not u0; Lagrange never reads it."""
    def response_to_w(basis_type):
        F, t, dt, u, w = source_problem()
        stepper = TimeStepper(F, tableau, t, dt, u, stage_type="value",
                              basis_type=basis_type, stage_functions=[w])
        fill(stepper.stage_functions[w], [0.0] * tableau.num_stages)

        w.assign(Constant(0.0))
        base = residual(stepper)
        w.assign(Constant(1.0))
        return np.linalg.norm(residual(stepper) - base)

    assert response_to_w("Bernstein") > 1e-12
    assert response_to_w(None) < 1e-14


@pytest.mark.parametrize("stage_type", stage_types)
def test_unregistered(stage_type):
    """Without stage_functions, w stays frozen and du/dt = 0."""
    F, t, dt, u, w = source_problem()
    stepper = TimeStepper(F, tableau, t, dt, u, stage_type=stage_type)

    assert stepper.stage_functions is None

    stepper.advance()
    assert abs(assemble(u * dx)) < 1e-14
