import pytest
import numpy as np
from firedrake import (inner, dx, UnitIntervalMesh, FunctionSpace,
                       assemble, TestFunctions, SpatialCoordinate,
                       project, as_vector, sin, pi, split)

from irksome import Dt, MeshConstant, TimeStepper, GaussLegendre
from irksome.tools import AI, IA

# test the energy conservation of the 1d wave equation in mixed form
# various time steppers.


def wave(n, deg, butcher_tableau, **kwargs):
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

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(2.0 / N)

    up = project(as_vector([0, sin(pi*x)]), Z)
    u, p = split(up)

    v, w = TestFunctions(Z)

    F = (inner(Dt(u), v)*dx + inner(u.dx(0), w) * dx
         + inner(Dt(p), w)*dx - inner(p, v.dx(0)) * dx)

    E = 0.5 * (inner(u, u)*dx + inner(p, p)*dx)

    stepper = TimeStepper(F, butcher_tableau, t, dt, up,
                          solver_parameters=params,
                          splitting=splitting)

    energies = []

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        energies.append(assemble(E))

    return np.array(energies)


@pytest.mark.parametrize('splitting', (AI, IA))
@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
@pytest.mark.parametrize(('deg', 'time_stages'),
                         [(1, i) for i in (1, 2)]
                         + [(2, i) for i in (2, 3)])
def test_wave_eq(deg, N, time_stages, splitting):
    energy = wave(N, deg, GaussLegendre(time_stages), splitting)
    print(energy)
    assert np.allclose(energy[1:], energy[:-1])


@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
@pytest.mark.parametrize('deg', (1, 2))
@pytest.mark.parameterize('pep', ((4, 2, 5),
                                  (5, 2, 6),
                                  (6, 3, 6),
                                  (7, 4, 6),
                                  (7, 5, 6)))
def test_pep_wave_eq(deg, N, time_stages, splitting):
    energy = wave(N, deg, PEPRK(time_stages), splitting)
    print(energy)
    assert np.allclose(energy[1:], energy[:-1])
