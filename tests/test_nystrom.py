from firedrake import (Constant, DirichletBC, Function, FunctionSpace, SpatialCoordinate,
                       TestFunction, UnitIntervalMesh, assemble, dx,
                       norm, grad, inner, pi, project, sin)
from irksome import Dt, GaussLegendre, StageDerivativeNystromTimeStepper


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
    dt = Constant(1.0 / N)

    uinit = sin(pi * x)

    butcher_tableau = GaussLegendre(time_stages)

    u0 = project(uinit, V)
    u = Function(u0)  # copy
    ut = Function(V)

    v = TestFunction(V)

    F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx

    bc = DirichletBC(V, 0, "on_boundary")

    E = 0.5 * inner(ut, ut) * dx + 0.5 * inner(grad(u), grad(u)) * dx

    stepper = StageDerivativeNystromTimeStepper(
        F, butcher_tableau, t, dt, u, ut, bcs=bc, solver_parameters=params)

    E0 = assemble(E)
    tf = 1
    while (float(t) < tf):
        if (float(t) + float(dt) > tf):
            dt.assign(tf - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return assemble(E) / E0, norm(u + u0)


def test_wave_eq():
    # number of refinements
    n = 5
    deg = 2
    stage_count = 2
    Erat, diff = wave(n, deg, stage_count)
    print(Erat, diff)
    assert abs(Erat - 1) < 1.e-8
    assert diff < 3.e-5
