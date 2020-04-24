import pytest
import numpy as np
from firedrake import (inner, dx, UnitIntervalMesh, FunctionSpace,
                       assemble, TestFunctions, SpatialCoordinate,
                       Constant, project, as_vector, sin, pi, split)

from irksome import Dt, TimeStepper, GaussLegendre

# test the energy conservation of the 1d wave equation in mixed form
# various time steppers.


def wave(n, deg, butcher_tableau):
    N = 2**n
    msh = UnitIntervalMesh(N)

    params = {"snes_type": "ksponly",
              "ksp_type": "preonly",
              "mat_type": "aij",
              "pc_type": "lu"}

    V = FunctionSpace(msh, "CG", deg)
    W = FunctionSpace(msh, "DG", deg - 1)
    Z = V * W

    x, = SpatialCoordinate(msh)

    t = Constant(0.0)
    dt = Constant(2.0 / N)

    up = project(as_vector([0, sin(pi*x)]), Z)
    u, p = split(up)

    v, w = TestFunctions(Z)

    F = (inner(Dt(u), v)*dx + inner(u.dx(0), w) * dx
         + inner(Dt(p), w)*dx - inner(p, v.dx(0)) * dx)

    E = 0.5 * (inner(u, u)*dx + inner(p, p)*dx)

    stepper = TimeStepper(F, butcher_tableau, t, dt, up,
                          solver_parameters=params)

    energies = []

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        energies.append(assemble(E))

    return np.array(energies)


@pytest.mark.parametrize(('deg', 'N', 'time_stages'),
                         [(1, 2**j, i) for j in range(2, 4)
                          for i in (1, 2)]
                         + [(2, 2**j, i) for j in range(2, 4)
                            for i in (2, 3)])
def test_wave_eq(deg, N, time_stages):
    energy = wave(N, deg, GaussLegendre(time_stages))
    print(energy)
    assert np.allclose(energy[1:], energy[:-1])
