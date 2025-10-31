import numpy as np
import pytest
from firedrake import (FunctionSpace, SpatialCoordinate, TestFunctions,
                       UnitIntervalMesh, as_vector, assemble, dx, inner, pi,
                       project, sin, split)
from irksome import PEPRK, Dt, GaussLegendre, TimeStepper
from irksome.tools import AI, IA

# test the energy conservation of the 1d wave equation in mixed form
# various time steppers.


def wave(n, deg, alpha, butcher_tableau, **kwargs):
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
    dt = Constant(alpha / N)

    up = project(as_vector([0, sin(pi*x)]), Z)
    u, p = split(up)

    v, w = TestFunctions(Z)

    F = (inner(Dt(u), v)*dx + inner(u.dx(0), w) * dx
         + inner(Dt(p), w)*dx - inner(p, v.dx(0)) * dx)

    E = 0.5 * (inner(u, u)*dx + inner(p, p)*dx)

    stepper = TimeStepper(F, butcher_tableau, t, dt, up,
                          solver_parameters=params,
                          **kwargs)

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
    kwargs = {"splitting": splitting}
    energy = wave(N, deg, 2.0, GaussLegendre(time_stages), **kwargs)
    assert np.allclose(energy[1:], energy[:-1])


@pytest.mark.parametrize('N', [2**j for j in range(2, 3)])
@pytest.mark.parametrize('deg', (1, 2))
@pytest.mark.parametrize('pep', ((4, 2, 5),
                                 (5, 2, 6),
                                 (6, 3, 6),
                                 (7, 4, 6),
                                 (7, 5, 6)))
def test_wave_eq_peprk(deg, N, pep):
    kwargs = {"stage_type": "explicit"}
    energy = wave(N, deg, 0.3, PEPRK(*pep), **kwargs)
    assert np.allclose(energy[1:], energy[:-1])
