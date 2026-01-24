import numpy as np
import pytest

from firedrake import (TestFunction, UnitSquareMesh, FunctionSpace, Function, grad, 
                       project, SpatialCoordinate, inner, dx, cos, pi, norm)
from irksome import (GaussLegendre, RadauIIA, Dt, MeshConstant, TimeStepper)
from FIAT import ufc_simplex
from FIAT.barycentric_interpolation import LagrangePolynomialSet
from FIAT.bernstein import Bernstein

import numpy as np

msh = UnitSquareMesh(20, 20)
params = {"snes_type": "newtonls", "snes_atol": 1.e-9}

def heat_value_hand(msh, tableau, dt_in, spatial_basis, spatial_degree, temporal_basis, temporal_degree, sample_points):

    V = FunctionSpace(msh, spatial_basis, spatial_degree)

    butcher_tableau = tableau(temporal_degree)

    MC = MeshConstant(msh)
    dt = MC.Constant(dt_in)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)

    u_init = 1 + cos(2*pi*x) * cos(2*pi*y)
    u = project(u_init, V)
    v = TestFunction(V)

    F = (inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx)

    kwargs = {"stage_type": "value",
              "basis_type": temporal_basis,
              "solver_parameters": params,
              }

    stepper = TimeStepper(F, butcher_tableau, t, dt, u, **kwargs)

    nodes = butcher_tableau.c
    nodes = np.insert(nodes, 0, 0.0)

    num_eval_points = len(sample_points)
    sample_points = np.reshape(sample_points, (-1, 1))

    if temporal_basis == "Lagrange":
        lag_basis = LagrangePolynomialSet(ufc_simplex(1), nodes)
        Vander_between = lag_basis.tabulate(sample_points, 0)[(0,)]

    elif temporal_basis == "Bernstein":
        bern_element = Bernstein(ufc_simplex(1), temporal_degree)
        Vander_between = bern_element.tabulate(0, sample_points)[(0,)]

    k0 = Function(V)
    k0.assign(u)
    stage_vals = k0.subfunctions + stepper.stages.subfunctions
    sample_values = Vander_between.T @ stage_vals

    u_interp = Function(V, name=f'u_interp')

    qs = []
    ts = []

    u_interp.interpolate(u)
    qs.append(u_interp.copy(deepcopy=True))
    ts.append(float(t))

    for i in range(2):
        
        stepper.advance()

        ##  Stage Interpolation
        for s in range(num_eval_points):
            u_interp.interpolate(sample_values[s])
            qs.append(u_interp.copy(deepcopy=True))
            ts.append(float(t) + sample_points[s] * float(dt))

        k0.assign(u)

        u_interp.interpolate(u)
        qs.append(u_interp.copy(deepcopy=True))       

        t.assign(float(t) + float(dt))
        ts.append(float(t))

    return (ts, qs)


def heat_value_mech(msh, tableau, dt_in, spatial_basis, spatial_degree, temporal_basis, temporal_degree, sample_points):

    V = FunctionSpace(msh, spatial_basis, spatial_degree)

    butcher_tableau = tableau(temporal_degree)

    MC = MeshConstant(msh)
    dt = MC.Constant(dt_in)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)
  
    u_init = 1 + cos(2*pi*x) * cos(2*pi*y)
    u = project(u_init, V)
    v = TestFunction(V)

    F = (inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx)

    kwargs = {"stage_type": "value",
              "basis_type": temporal_basis,
              "solver_parameters": params,
              "sample_points": sample_points
              }

    stepper = TimeStepper(F, butcher_tableau, t, dt, u, **kwargs)

    qs = []
    ts = []

    u_interp = u.copy(deepcopy=True)

    qs = [u_interp.copy(deepcopy=True)]
    ts.append(float(t))

    for i in range(2):

        stepper.advance()

        for (i, val) in enumerate(stepper.sample_values):
            ts.append(float(t) + sample_points[i] * float(dt))
            u_interp.interpolate(val)
            qs.append(u_interp.copy(deepcopy=True))

        qs.append(u.copy(deepcopy=True))
      
        t.assign(float(t) + float(dt))
        ts.append(float(t))
            
    return (ts, qs)


def heat_deriv_mech(msh, tableau, dt_in, spatial_basis, spatial_degree, temporal_degree, sample_points):

    V = FunctionSpace(msh, spatial_basis, spatial_degree)

    butcher_tableau = tableau(temporal_degree)

    MC = MeshConstant(msh)
    dt = MC.Constant(dt_in)
    t = MC.Constant(0.0)

    x, y = SpatialCoordinate(msh)
  
    u_init = 1 + cos(2*pi*x) * cos(2*pi*y)
    u = project(u_init, V)
    v = TestFunction(V)

    F = (inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx)

    kwargs = {"stage_type": "deriv",
              "solver_parameters": params,
              "sample_points": sample_points
              }

    stepper = TimeStepper(F, butcher_tableau, t, dt, u, **kwargs)

    qs = []
    ts = []

    u_interp = u.copy(deepcopy=True)

    qs = [u_interp.copy(deepcopy=True)]
    ts.append(float(t))

    for i in range(2):

        stepper.advance()

        for (i, val) in enumerate(stepper.sample_values):
            ts.append(float(t) + sample_points[i] * float(dt))
            u_interp.interpolate(val)
            qs.append(u_interp.copy(deepcopy=True))

        qs.append(u.copy(deepcopy=True))
        t.assign(float(t) + float(dt))
        ts.append(float(t))

    return (ts, qs)

dt_in = 0.125
sample_points = [0.2, 0.5, 0.75, 0.9]

@pytest.mark.parametrize('tableau', [RadauIIA, GaussLegendre])
@pytest.mark.parametrize('spatial_degree', [1, 2])
@pytest.mark.parametrize('temporal_basis', ['Bernstein', 'Lagrange'])
@pytest.mark.parametrize('temporal_degree', [1, 2, 3])
def test_stage_value(tableau, spatial_degree, temporal_basis, temporal_degree):
    ts_hand, qs_hand =   heat_value_hand(msh, tableau, 0.125, 'Lagrange', spatial_degree, temporal_basis, temporal_degree, sample_points)
    ts_stage, qs_stage = heat_value_mech(msh, tableau, 0.125, 'Lagrange', spatial_degree, temporal_basis, temporal_degree, sample_points)
    errors = [norm(qs_hand[i] - qs_stage[i]) for i in range(len(qs_hand))]
    assert max(errors) < 1e-11

@pytest.mark.parametrize('tableau', [RadauIIA, GaussLegendre])
@pytest.mark.parametrize('spatial_degree', [1, 2])
@pytest.mark.parametrize('temporal_degree', [1, 2, 3])
def test_stage_deriv(tableau, spatial_degree, temporal_degree):
    ts_hand, qs_hand =   heat_value_hand(msh, tableau, 0.125, 'Lagrange', spatial_degree, 'Lagrange', temporal_degree, sample_points)
    ts_deriv, qs_deriv = heat_deriv_mech(msh, tableau, 0.125, 'Lagrange', spatial_degree, temporal_degree, sample_points)
    errors = [norm(qs_hand[i] - qs_deriv[i]) for i in range(len(qs_hand))]
    assert max(errors) < 1e-11
