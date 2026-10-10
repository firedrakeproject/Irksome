import numpy as np
import pytest
from firedrake import *
from irksome import PEPRK, Dt, MeshConstant, TimeStepper, SSPButcherTableau

peprks = [PEPRK(*x) for x in ((4, 2, 5), (5, 2, 6))]
ssprks = [SSPButcherTableau(2, 2), SSPButcherTableau(2, 3), SSPButcherTableau(3, 3)]

bt_list = peprks + ssprks
id_list = ["PEP(4,2,5)", "PEP(5,2,6)", "SSP(2,2)", "SSP(2,3)", "SSP(3,3)"]

L = 10.0


@pytest.fixture
def msh():
    return IntervalMesh(10, L)


def run_1d_heat(butcher_tableau, V, nsteps_per_unit_time, t_end):
    MC = MeshConstant(V.mesh())
    dt = MC.Constant(1.0 / nsteps_per_unit_time)
    t = MC.Constant(0.0)
    (x,) = SpatialCoordinate(V.mesh())

    # Boundary values
    u_0 = Constant(2.0)
    u_1 = Constant(3.0) + atan(t)

    # Method of manufactured solutions.  Taking a solution that is linear in
    # x makes it exactly representable in V, so that the error against it is
    # purely temporal and reports the order of the time stepper.
    uexact = u_0 + (x / L) * (u_1 - u_0)
    rhs = Dt(uexact) - div(grad(uexact))
    u = Function(V)
    u.interpolate(uexact)
    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(rhs, v) * dx
    )
    bcs = [
        DirichletBC(V, u_1, 2),
        DirichletBC(V, u_0, 1),
    ]

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = TimeStepper(
        F, butcher_tableau, t, dt, u, bcs=bcs,
        solver_parameters=luparams,
        stage_type="explicit"
    )

    bnd_error = inner(u-uexact, u-uexact) * ds
    for _ in range(round(t_end * nsteps_per_unit_time)):
        stepper.advance()
        t.assign(float(t) + float(dt))
        # The stage boundary data is imposed exactly at every step
        assert abs(assemble(bnd_error)) ** 0.5 < 1e-12
    return errornorm(uexact, u)


# Note that this test is constructed with dt small enough relative to
# dx that these explicit methods stay stable -- while Irksome provides
# support for explicit schemes, we also caution users that there are
# no checks in the code that the method you are trying to run is
# actually sensible!
@pytest.mark.parametrize("butcher_tableau", bt_list, ids=id_list)
def test_1d_heat_dirichletbc(butcher_tableau, msh):
    nsteps_per_unit_time = 10
    t_end = 2.0
    V = FunctionSpace(msh, "CG", 1)

    errors = np.array([run_1d_heat(butcher_tableau, V,
                                   (2**r) * nsteps_per_unit_time, t_end)
                       for r in range(3)])
    rates = np.diff(-np.log2(errors))
    assert (rates > butcher_tableau.order - 0.25).all()
