import numpy as np
import pytest
from firedrake import (FacetNormal, Function, FunctionSpace, SpatialCoordinate,
                       TestFunctions, UnitSquareMesh, cos, diff, div, dot, ds,
                       dx, errornorm, exp, grad, inner, norm, pi, split)
from irksome import Dt, MeshConstant, RadauIIA, TimeStepper
from ufl.algorithms import expand_derivatives

lu_params = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "mat_type": "aij",
    "pc_type": "lu"
}

vi_params = {
    "snes_type": "vinewtonrsls",
    "snes_vi_monitor": None,
    "snes_max_it": 300,
    "snes_atol": 1.e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
}


def mixed_heat(n, deg, butcher_tableau, solver_parameters,
               **kwargs):
    N = 2**n
    msh = UnitSquareMesh(N, N)

    V = FunctionSpace(msh, "RT", deg)
    W = FunctionSpace(msh, "DG", deg-1)

    Z = V * W

    x, y = SpatialCoordinate(msh)

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(1.0 / N)

    pexact = (cos(pi * x) * cos(pi * y))**2 * exp(-t)
    uexact = -grad(pexact)

    up = Function(Z)
    u, p = split(up)

    v, w = TestFunctions(Z)

    n = FacetNormal(msh)

    rhs = expand_derivatives(diff(pexact, t) + div(uexact))

    F = (inner(Dt(p), w) * dx
         + inner(div(u), w) * dx
         - inner(rhs, w) * dx
         + inner(u, v) * dx
         - inner(p, div(v)) * dx
         + inner(pexact, dot(v, n)) * ds)

    up.subfunctions[1].project(pexact)

    stepper = TimeStepper(F, butcher_tableau, t, dt, up,
                          solver_parameters=solver_parameters,
                          **kwargs)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    u, p = up.subfunctions
    erru = errornorm(uexact, u) / norm(uexact)
    errp = errornorm(pexact, p) / norm(pexact)
    return erru + errp


@pytest.mark.parametrize('butcher_tableau', [RadauIIA(i) for i in (1, 2)])
def test_heat_bern(butcher_tableau):
    deg = 1
    kwargs = {"stage_type": "value",
              "basis_type": "Bernstein",
              "solver_parameters": lu_params}
    diff = np.array([mixed_heat(i, deg, butcher_tableau, **kwargs) for i in range(2, 4)])
    print(diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print(conv)
    assert (conv > (deg-0.1)).all()


@pytest.mark.parametrize('butcher_tableau', [RadauIIA(i) for i in (1, 2)])
@pytest.mark.parametrize('bounds_type', ('stage', 'last_stage', 'time_level'))
def test_heat_bern_bounds(butcher_tableau, bounds_type):
    deg = 1
    bounds = (bounds_type, (None, 0), (None, None))
    kwargs = {"stage_type": "value",
              "basis_type": "Bernstein",
              "bounds": bounds,
              "solver_parameters": vi_params}
    diff = np.array([mixed_heat(i, deg, butcher_tableau, **kwargs) for i in range(2, 4)])
    print(diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print(conv)
    assert (conv > (deg-0.1)).all()
