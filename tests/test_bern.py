import numpy as np
import pytest
from firedrake import (FacetNormal, Function, FunctionSpace, SpatialCoordinate,
                       TestFunctions, UnitSquareMesh, cos, diff, div, dot, ds,
                       dx, errornorm, exp, grad, inner, norm, pi, sin, split)
from irksome import Dt, MeshConstant, RadauIIA, TimeStepper
from ufl.algorithms import expand_derivatives


def heat(n, deg, butcher_tableau, **kwargs):
    N = 2**n
    msh = UnitSquareMesh(N, N)

    params = {"snes_type": "ksponly",
              "ksp_type": "preonly",
              "mat_type": "aij",
              "pc_type": "lu"}

    V = FunctionSpace(msh, "RT", deg)
    W = FunctionSpace(msh, "DG", deg-1)

    Z = V * W

    x, y = SpatialCoordinate(msh)

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(1.0 / N)

    pexact = sin(pi * x) * cos(2 * pi * y) * exp(-t)
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
                          solver_parameters=params,
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
              "basis_type": "Bernstein"}
    diff = np.array([heat(i, deg, butcher_tableau, **kwargs) for i in range(2, 4)])
    print(diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print(conv)
    assert (conv > (deg-0.1)).all()
