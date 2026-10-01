"""``stage_functions`` substitutes a registered coefficient stage by stage.

The model problem is du/dt = w(t) with w(t) = t, exact after one step at
dt**2/2.  With w frozen the step returns 0, so the accuracy check has teeth.
"""
import numpy as np
import pytest
from firedrake import (Constant, Function, FunctionSpace, TestFunction,
                       UnitIntervalMesh, assemble, dx, inner)
from irksome import Dt, MeshConstant, RadauIIA, TimeStepper

stage_types = ("deriv", "value")
tableau = RadauIIA(2)
dt_value = 0.25


def source_problem():
    """du/dt = w, with w an ordinary coefficient of the form."""
    msh = UnitIntervalMesh(2)
    V = FunctionSpace(msh, "DG", 0)
    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(dt_value)
    u = Function(V)
    w = Function(V)
    v = TestFunction(V)
    F = inner(Dt(u), v) * dx - inner(w, v) * dx
    return F, t, dt, u, w


def stage_dofs(W):
    return np.concatenate([np.asarray(Wi.dat.data_ro).ravel()
                           for Wi in W.subfunctions])


def residual(stepper):
    """Assembled stage residual, one row per stage."""
    Fbig, _ = stepper.get_form_and_bcs(stepper.stages)
    r = assemble(Fbig)
    return np.array([np.asarray(ri.dat.data_ro).ravel()
                     for ri in r.subfunctions])


def fill(W, values):
    for Wi, val in zip(W.subfunctions, values):
        Wi.assign(Constant(val))


@pytest.mark.parametrize("stage_type", stage_types)
def test_stage_function_is_substituted_stage_by_stage(stage_type):
    """Each stage of the registered coefficient moves the residual on its own.

    Perturbing one stage of ``w`` and no other must change the residual, and
    the change must differ from stage to stage.  A coefficient substituted
    once for the whole step would give either no response or the same response
    to every stage.
    """
    F, t, dt, u, w = source_problem()
    stepper = TimeStepper(F, tableau, t, dt, u,
                          stage_type=stage_type, stage_functions=[w])

    W = stepper.stage_functions[w]
    num_stages = tableau.num_stages
    assert len(W.subfunctions) == num_stages

    fill(W, [0.0] * num_stages)
    base = residual(stepper)

    columns = []
    for j in range(num_stages):
        fill(W, [1.0 if i == j else 0.0 for i in range(num_stages)])
        columns.append(residual(stepper) - base)

    for j, col in enumerate(columns):
        print(f"{stage_type}: response to stage {j} = {col.ravel()}")
        assert np.linalg.norm(col) > 1e-12

    for j in range(num_stages):
        for k in range(j + 1, num_stages):
            assert np.linalg.norm(columns[j] - columns[k]) > 1e-12


@pytest.mark.parametrize("stage_type", stage_types)
def test_stage_function_carries_a_time_dependent_source(stage_type):
    """Filling the stage function with w(t) = t integrates du/dt = t exactly.

    The value formulation solves for the stage values themselves, so ``w``
    takes ``w(t_n + c_i*dt) = c_i*dt``.  The derivative formulation
    reconstructs ``w_i = w + dt*sum_j a_ij k_j``, and the row sums of ``A``
    are ``c``, so every stage derivative of ``w`` is ``dw/dt = 1``.
    """
    F, t, dt, u, w = source_problem()
    stepper = TimeStepper(F, tableau, t, dt, u,
                          stage_type=stage_type, stage_functions=[w])

    W = stepper.stage_functions[w]
    if stage_type == "value":
        fill(W, [float(ci) * dt_value for ci in tableau.c])
    else:
        fill(W, [1.0] * tableau.num_stages)
    print(f"{stage_type}: stage function dofs = {stage_dofs(W)}")

    stepper.advance()

    got = assemble(u * dx)
    exact = dt_value ** 2 / 2
    print(f"{stage_type}: u(dt) = {got:.12f}, exact = {exact:.12f}")
    assert abs(got - exact) < 1e-12


def test_registered_coefficient_supplies_its_own_value_at_t_n():
    """``to_value`` prepends the coefficient's own value, not ``u0``'s.

    Only a non-trivial Vandermonde reaches that value, so Lagrange is the
    control: there w is replaced outright and the residual cannot see it.
    """
    def response_to_w(basis_type):
        F, t, dt, u, w = source_problem()
        stepper = TimeStepper(F, tableau, t, dt, u, stage_type="value",
                              basis_type=basis_type, stage_functions=[w])
        fill(stepper.stage_functions[w], [0.0] * tableau.num_stages)

        w.assign(Constant(0.0))
        base = residual(stepper)
        w.assign(Constant(1.0))
        return np.linalg.norm(residual(stepper) - base)

    bernstein = response_to_w("Bernstein")
    lagrange = response_to_w(None)
    print(f"response to w: Bernstein {bernstein:.3e}, Lagrange {lagrange:.3e}")

    assert bernstein > 1e-12
    assert lagrange < 1e-14


@pytest.mark.parametrize("stage_type", stage_types)
def test_unregistered_coefficient_stays_frozen(stage_type):
    """The control: without the kwarg there is no per-stage handle at all.

    ``w`` keeps its single value through the step, so the same problem
    integrates ``du/dt = 0`` and returns u(0).
    """
    F, t, dt, u, w = source_problem()
    stepper = TimeStepper(F, tableau, t, dt, u, stage_type=stage_type)

    assert stepper.stage_functions is None

    stepper.advance()
    assert abs(assemble(u * dx)) < 1e-14
