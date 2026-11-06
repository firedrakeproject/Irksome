from firedrake import *
import numpy as np
from irksome import (Dt, TimeStepper, BackwardEuler, LobattoIIIA,
                     MeshConstant, RadauIIA)
import pytest


@pytest.mark.parametrize("basis_type", ["Lagrange", "Bernstein"])
@pytest.mark.parametrize("degree", [1, 2])
def test_stage_value_init(basis_type, degree):
    nx = 16
    lx = 1.0
    mesh = IntervalMesh(nx, lx)

    Q = FunctionSpace(mesh, "DG", 0)
    x, = SpatialCoordinate(mesh)

    p = Function(Q)
    p_in = Constant(0.5)
    p.assign(p_in)

    q = TestFunction(Q)

    u_0 = Constant(1.0)
    du = Constant(1.0)
    Lx = Constant(lx)
    u = as_vector((u_0 + du * x / Lx,))

    p_max = Constant(2.0)
    s = p_max / p - 1
    F_cells = (Dt(p) * q - inner(p * u, grad(q)) - s * q) * dx
    n = FacetNormal(mesh)
    f_n = p * max_value(0, inner(u, n))
    F_facets = (f_n("+") - f_n("-")) * (q("+") - q("-")) * dS
    F_inflow = p_in * min_value(0, inner(u, n)) * q * ds
    F_outflow = p * max_value(0, inner(u, n)) * q * ds
    F = F_cells + F_facets + F_inflow + F_outflow

    method = BackwardEuler() if degree == 1 else RadauIIA(degree)
    t = Constant(0.0)
    timestep = 0.5 / nx
    dt = Constant(timestep)
    params = {
        "stage_type": "value",
        "basis_type": basis_type,
        "solver_parameters": {"snes_monitor": None},
    }
    stepper = TimeStepper(F, method, t, dt, p, **params)

    final_time = 2.0
    num_steps = int(final_time / timestep)
    for step in range(num_steps):
        stepper.advance()

    assert norm(p) > 0.0


@pytest.mark.parametrize("butcher_tableau,expected_solved", [(LobattoIIIA(2), 1),
                                                             (LobattoIIIA(3), 2),
                                                             (RadauIIA(2), 2)])
def test_explicit_first_stage_is_not_solved_for(butcher_tableau, expected_solved):
    """An explicit first stage takes u0, so it is spliced in rather than solved.

    The stage system shrinks by one stage and the method keeps its order.
    """
    msh = UnitIntervalMesh(8)
    V = FunctionSpace(msh, "CG", 2)
    (x,) = SpatialCoordinate(msh)
    t_end = 1.0

    def error(nsteps):
        MC = MeshConstant(msh)
        t = MC.Constant(0.0)
        dt = MC.Constant(t_end / nsteps)
        uexact = 2.0 + x * (1.0 + atan(t)) + x * (1 - x) * cos(t)
        rhs = Dt(uexact) - div(grad(uexact))
        u = Function(V).interpolate(uexact)
        v = TestFunction(V)
        F = (inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
             - inner(rhs, v) * dx)
        stepper = TimeStepper(F, butcher_tableau, t, dt, u,
                              bcs=DirichletBC(V, uexact, "on_boundary"),
                              stage_type="value")
        assert stepper.num_solved_stages == expected_solved
        assert (stepper.stages.function_space().dim()
                == expected_solved * V.dim())
        for _ in range(nsteps):
            stepper.advance()
            t.assign(float(t) + float(dt))
        return errornorm(uexact, u)

    errors = np.array([error(10 * 2**r) for r in range(3)])
    rates = np.diff(-np.log2(errors))
    assert (rates > butcher_tableau.order - 0.5).all()


@pytest.mark.parametrize("kwargs", [
    {"nullspace": VectorSpaceBasis(constant=True)},
    {"near_nullspace": VectorSpaceBasis(constant=True)},
    {"transpose_nullspace": VectorSpaceBasis(constant=True)},
    {"bounds": ("stage", Constant(-1.0), Constant(2.0)),
     "solver_parameters": {"snes_type": "vinewtonrsls"}},
    {"bounds": ("last_stage", Constant(-1.0), None),
     "solver_parameters": {"snes_type": "vinewtonrsls"}}])
def test_explicit_first_stage_with_solver_options(kwargs):
    msh = UnitIntervalMesh(8)
    V = FunctionSpace(msh, "CG", 1)
    (x,) = SpatialCoordinate(msh)
    t = Constant(0.0)
    dt = Constant(0.1)
    u = Function(V).interpolate(cos(pi * x))
    uplain = u.copy(deepcopy=True)
    v = TestFunction(V)
    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
    Fplain = replace(F, {u: uplain})
    TimeStepper(Fplain, LobattoIIIA(3), t, dt, uplain, stage_type="value").advance()
    kwargs = dict(kwargs)
    if "bounds" in kwargs:
        kind, lower, upper = kwargs["bounds"]
        lower = Function(V).assign(lower)
        upper = upper if upper is None else Function(V).assign(upper)
        kwargs["bounds"] = (kind, lower, upper)
    stepper = TimeStepper(F, LobattoIIIA(3), t, dt, u, stage_type="value", **kwargs)
    stepper.advance()
    assert stepper.num_solved_stages == 2
    assert errornorm(uplain, u) < 1e-8
