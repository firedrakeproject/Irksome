import numpy as np
import pytest
from firedrake import (Constant, DirichletBC, Function, FunctionSpace, SpatialCoordinate,
                       TestFunction, UnitIntervalMesh, cos, diff, div, dx,
                       errornorm, exp, grad, inner, norm, pi, project, sin)
from irksome import Dt, GaussLegendre, TimeStepper, StageDerivativeNystromTimeStepper
from ufl.algorithms import expand_derivatives


def wave(n, deg, time_stages):
    N = 2**n
    msh = UnitIntervalMesh(N)

    params = {"snes_type": "ksponly",
              "ksp_type": "preonly",
              "mat_type": "aij",
              "pc_type": "lu"}

    V = FunctionSpace(msh, "CG", deg)
    x, = SpatialCoordinate(msh)

    t = Constant(0.0)
    dt = Constant(2.0 / N)

    uexact = sin(pi * x) * cos(pi * t)

    butcher_tableau = GaussLegendre(time_stages)

    u0 = project(uexact, V)
    u = Function(u0)  # copy
    ut = Function(V)
    
    v = TestFunction(V)

    F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx

    bc = DirichletBC(V, 0, "on_boundary")

    E = 0.5 * inner(ut, ut) * dx + 0.5 * inner(grad(u), grad(u)) * dx
    
    stepper = StageDerivativeNystromTimeStepper(
        F, butcher_tableau, t, dt, u, ut,
        bcs=bc, solver_parameters=params)

    E0 = assemble(E)
    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return assemble(E) / E0, errornorm(u, u0)


def test_wave_eq():
    # number of refinements
    n = 3
    deg = 1
    stage_count = 1
    Erat, diff = wave(n, deg, stage_count)
    print(Erat)
    print(diff)

    
