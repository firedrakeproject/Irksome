from math import isclose

import pytest
from firedrake import *
from irksome import Dt, MeshConstant, TimeStepper, IMEXEuler, IMEX2, IMEX3, IMEX4
from ufl.algorithms.ad import expand_derivatives


def convdiff_neumannbc(butcher_tableau, order, N):
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", order)
    MC = MeshConstant(msh)
    dt = MC.Constant(0.1 / N)
    t = MC.Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    # Choose uexact so rhs is nonzero
    uexact = cos(pi*x)*exp(-t)
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) - uexact.dx(0)
    u = Function(V)
    u.interpolate(uexact)

    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(rhs, v) * dx
    )
    Fexp = inner(u.dx(0), v)*dx

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = TimeStepper(
        F, butcher_tableau, t, dt, u, Fexp=Fexp,
        solver_parameters=luparams, mass_parameters=luparams,
        stage_type="dirkimex"
    )

    t_end = 0.1
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return (errornorm(uexact, u) / norm(uexact))


@pytest.mark.parametrize("butcher_tableau, order",
                         [(IMEXEuler(), 1), (IMEX2(), 2),
                          (IMEX3(), 3), (IMEX4(), 3)])
def test_1d_convdiff_neumannbc(butcher_tableau, order):
    errs = np.array([convdiff_neumannbc(butcher_tableau, order, 10*2**p) for p in [3, 4]])
    print(errs)
    conv = np.log2(errs[0]/errs[1])
    print(conv)
    assert conv > order-0.4


# Note IMEX4 is stiffly accurate, so satisfies BC checks.  IMEX2 and IMEX3 do not
@pytest.mark.parametrize("butcher_tableau", [IMEXEuler(), IMEX4()])
def test_1d_heat_dirichletbc(butcher_tableau):
    # Boundary values
    u_0 = Constant(2.0)
    u_1 = Constant(3.0)

    N = 10
    x0 = 0.0
    x1 = 10.0
    msh = IntervalMesh(N, x1)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    # Method of manufactured solutions copied from Heat equation demo.
    S = Constant(2.0)
    C = Constant(1000.0)
    B = (x - Constant(x0)) * (x - Constant(x1)) / C
    R = (x * x) ** 0.5
    # Note end linear contribution
    uexact = (
        B * atan(t) * (pi / 2.0 - atan(S * (R - t)))
        + u_0
        + ((x - x0) / x1) * (u_1 - u_0)
    )
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))
    u = Function(V)
    u.interpolate(uexact)
    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
    )
    Fexp = inner(rhs, v) * dx

    bc = [
        DirichletBC(V, u_1, 2),
        DirichletBC(V, u_0, 1),
    ]

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = TimeStepper(
        F, butcher_tableau, t, dt, u, Fexp=Fexp, bcs=bc,
        solver_parameters=luparams, mass_parameters=luparams,
        stage_type="dirkimex"
    )

    t_end = 2.0
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        # Check solution and boundary values
        assert errornorm(uexact, u) / norm(uexact) < 10.0 ** -3
        assert isclose(u.at(x0), u_0)
        assert isclose(u.at(x1), u_1)
