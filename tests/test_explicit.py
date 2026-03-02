from math import isclose

import pytest
from firedrake import *
from irksome import PEPRK, Dt, MeshConstant, TimeStepper, SSPButcherTableau

peprks = [PEPRK(*x) for x in ((4, 2, 5), (5, 2, 6))]
ssprks = [SSPButcherTableau(2, 2), SSPButcherTableau(2, 3), SSPButcherTableau(3, 3)]

bt_list = peprks + ssprks
id_list = ["PEP(4,2,5)", "PEP(5,2,6)", "SSP(2,2)", "SSP(2,3)", "SSP(3,3)"]


# Note that this test is constructed with dt small enough relative to
# dx that these explicit methods stay stable -- while Irksome provides
# support for explicit schemes, we also caution users that there are
# no checks in the code that the method you are trying to run is
# actually sensible!
@pytest.mark.parametrize("butcher_tableau", bt_list, ids=id_list)
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
    rhs = Dt(uexact) - div(grad(uexact))
    u = Function(V)
    u.interpolate(uexact)
    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(rhs, v) * dx
    )
    bc = [
        DirichletBC(V, u_1, 2),
        DirichletBC(V, u_0, 1),
    ]

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = TimeStepper(
        F, butcher_tableau, t, dt, u, bcs=bc,
        solver_parameters=luparams,
        stage_type="explicit"
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
