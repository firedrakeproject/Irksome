import numpy as np
import pytest

from firedrake import (Constant, TestFunction, UnitSquareMesh, FunctionSpace, Function, grad,
                       project, SpatialCoordinate, inner, dx, cos, pi, norm,
                       as_vector, DirichletBC, div, dot, ds, errornorm, FacetNormal, split, TestFunctions)
from irksome import (GaussLegendre, RadauIIA, Dt, TimeStepper)
from irksome.tools import replace
from FIAT import ufc_simplex
from FIAT.barycentric_interpolation import LagrangePolynomialSet
from FIAT.bernstein import Bernstein

params = {"snes_type": "ksponly", "ksp_type": "preonly", "pc_type": "lu"}


@pytest.fixture
def msh():
    return UnitSquareMesh(20, 20)


@pytest.fixture(params=[1, 2])
def V(msh, request):
    degree = request.param
    return FunctionSpace(msh, "Lagrange", degree)


def heat_stepper(V, butcher_tableau, dt_in, **kwargs):
    dt = Constant(dt_in)
    t = Constant(0.0)

    x, y = SpatialCoordinate(V.mesh())

    u_init = 1 + cos(2*pi*x) * cos(2*pi*y)
    u = project(u_init, V)
    v = TestFunction(V)

    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx

    stepper = TimeStepper(F, butcher_tableau, t, dt, u, **kwargs)
    return stepper


def heat_value_hand(V, butcher_tableau, dt_in, **kwargs):
    sample_points = kwargs.pop("sample_points")
    temporal_basis = kwargs.get("basis_type", "Lagrange")
    kwargs["stage_type"] = "value"

    stepper = heat_stepper(V, butcher_tableau, dt_in, **kwargs)
    u = stepper.u0
    t = stepper.t
    dt = stepper.dt

    nodes = butcher_tableau.c
    nodes = np.insert(nodes, 0, 0.0)

    num_eval_points = len(sample_points)
    sample_points = np.reshape(sample_points, (-1, 1))

    if temporal_basis == "Lagrange":
        lag_basis = LagrangePolynomialSet(ufc_simplex(1), nodes)
        Vander_between = lag_basis.tabulate(sample_points, 0)[(0,)]

    elif temporal_basis == "Bernstein":
        bern_element = Bernstein(ufc_simplex(1), len(nodes)-1)
        Vander_between = bern_element.tabulate(0, sample_points)[(0,)]

    u_old = Function(u)
    stage_vals = u_old.subfunctions + stepper.stages.subfunctions
    sample_values = Vander_between.T @ stage_vals

    u_interp = Function(u, name='u_interp')
    qs = [u_interp.copy(deepcopy=True)]
    ts = [float(t)]

    for step in range(2):
        u_old.assign(u)

        stepper.advance()

        # Stage Interpolation
        for point, val in zip(sample_points, sample_values):
            u_interp.assign(val)
            ts.append(float(t) + point * float(dt))
            qs.append(u_interp.copy(deepcopy=True))

        t.assign(t + dt)
        ts.append(float(t))
        qs.append(u.copy(deepcopy=True))

    return (ts, qs)


def heat_mech(V, butcher_tableau, dt_in, **kwargs):

    stepper = heat_stepper(V, butcher_tableau, dt_in, **kwargs)
    u = stepper.u0
    t = stepper.t
    dt = stepper.dt

    u_interp = Function(u, name='u_interp')
    qs = [u_interp.copy(deepcopy=True)]
    ts = [float(t)]

    for step in range(2):

        stepper.advance()

        for point, val in zip(stepper.sample_points, stepper.sample_values):
            u_interp.assign(val)
            ts.append(float(t) + point * float(dt))
            qs.append(u_interp.copy(deepcopy=True))

        t.assign(t + dt)
        ts.append(float(t))
        qs.append(u.copy(deepcopy=True))

    return (ts, qs)


@pytest.mark.parametrize('scheme', [RadauIIA, GaussLegendre])
@pytest.mark.parametrize('temporal_degree', [1, 3])
def test_sample_Bernstein(V, scheme, temporal_degree):
    kwargs = dict(
        stage_type="value",
        basis_type="Bernstein",
        solver_parameters=params,
        sample_points=[0.2, 0.5, 0.75, 0.9],
    )
    tableau = scheme(temporal_degree)
    dt_in = 0.125

    ts_hand, qs_hand = heat_value_hand(V, tableau, dt_in, **kwargs)
    ts_mech, qs_mech = heat_mech(V, tableau, dt_in, **kwargs)
    errors = [norm(qs_hand[i] - qs_mech[i]) for i in range(len(qs_hand))]
    assert max(errors) < 1e-11


@pytest.mark.parametrize('scheme', [RadauIIA, GaussLegendre])
@pytest.mark.parametrize('temporal_degree', [1, 3])
@pytest.mark.parametrize('stage_type', ["value", "deriv"])
def test_sample_Lagrange(V, scheme, temporal_degree, stage_type):
    kwargs = dict(
        stage_type=stage_type,
        solver_parameters=params,
        sample_points=[0.2, 0.5, 0.75, 0.9],
    )
    tableau = scheme(temporal_degree)
    dt_in = 0.125

    ts_hand, qs_hand = heat_value_hand(V, tableau, dt_in, **kwargs)
    ts_mech, qs_mech = heat_mech(V, tableau, dt_in, **kwargs)
    errors = [norm(qs_hand[i] - qs_mech[i]) for i in range(len(qs_hand))]
    assert max(errors) < 1e-11


@pytest.mark.parametrize('scheme', [RadauIIA, GaussLegendre])
@pytest.mark.parametrize('temporal_degree', [2, 3])
@pytest.mark.parametrize('stage_type', ["deriv", "value"])
def test_mixed_heat(scheme, temporal_degree, stage_type):

    msh = UnitSquareMesh(4, 4)
    V = FunctionSpace(msh, "RT", 3)
    W = FunctionSpace(msh, "DG", 2)
    Z = V * W

    dt = Constant(0.05)
    t = Constant(0.0)
    my_t = Constant(0.0)

    x, y = SpatialCoordinate(msh)

    u_exact = (x*(1-x)+y*(1-y))*(1+t)
    q_exact = -grad(u_exact)
    my_qu = replace(as_vector([q_exact[0], q_exact[1], u_exact]), {t: my_t})
    f = Dt(u_exact) + div(q_exact)

    qu = project(my_qu, Z)
    q, u = split(qu)

    v, w = TestFunctions(Z)
    n = FacetNormal(msh)

    F = (inner(Dt(u), w) * dx + inner(div(q), w) * dx - inner(f, w) * dx
         + inner(q, v) * dx - inner(u, div(v)) * dx - inner(u_exact, dot(v, n))*ds)

    bc = DirichletBC(Z.sub(0), q_exact, "on_boundary")

    sample_points = [0.3, 0.75]
    stepper = TimeStepper(F, scheme(temporal_degree), t, dt, qu, bcs=bc,
                          stage_type=stage_type, sample_points=sample_points)

    e_vals = []
    check = Function(qu)
    for step in range(2):

        stepper.advance()

        for (i, val) in enumerate(stepper.sample_values):
            my_t.assign(float(t) + sample_points[i] * float(dt))
            check.assign(val)
            e_vals.append(errornorm(my_qu, check))

        t.assign(float(t)+float(dt))

    assert np.allclose(e_vals, 0)
