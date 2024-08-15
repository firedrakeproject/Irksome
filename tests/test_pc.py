import numpy
import pytest
from firedrake import (DirichletBC, Function, FunctionSpace, SpatialCoordinate,
                       TestFunction, UnitSquareMesh, diff, div, dx, errornorm,
                       grad, inner)
from irksome import Dt, MeshConstant, LobattoIIIC, RadauIIA, TimeStepper
from ufl.algorithms.ad import expand_derivatives

# Tests that various PCs are actually getting the right answer.


def Fubc(V, uexact, rhs):
    u = Function(V)
    u.interpolate(uexact)
    v = TestFunction(V)
    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx

    bc = DirichletBC(V, uexact, "on_boundary")

    return (F, u, bc)


def heat(butcher_tableau):
    N = 4
    msh = UnitSquareMesh(N, N)

    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)

    deg = 2
    V = FunctionSpace(msh, "CG", deg)

    x, y = SpatialCoordinate(msh)

    uexact = t*(x+y)
    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    sols = []

    luparams = {"mat_type": "aij",
                "snes_type": "ksponly",
                "ksp_type": "preonly",
                "pc_type": "lu"}

    ranaLD = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "gmres",
              "ksp_monitor": None,
              "pc_type": "python",
              "pc_python_type": "irksome.RanaLD",
              "aux": {
                  "pc_type": "fieldsplit",
                  "pc_fieldsplit_type": "multiplicative"
              }}
    per_field = {"ksp_type": "preonly",
                 "pc_type": "gamg"}

    for s in range(butcher_tableau.num_stages):
        ranaLD["fieldsplit_%s" % (s,)] = per_field

    ranaDU = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "gmres",
              "ksp_monitor": None,
              "pc_type": "python",
              "pc_python_type": "irksome.RanaDU",
              "aux": {
                  "pc_type": "fieldsplit",
                  "pc_fieldsplit_type": "multiplicative"
              }}

    for s in range(butcher_tableau.num_stages):
        ranaDU["fieldsplit_%s" % (s,)] = per_field

    params = [luparams, ranaLD, ranaDU]

    for solver_parameters in params:
        F, u, bc = Fubc(V, uexact, rhs)

        stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                              solver_parameters=solver_parameters)
        stepper.advance()
        sols.append(u)

    errs = [errornorm(sols[0], uu) for uu in sols[1:]]
    return numpy.max(errs)


@pytest.mark.parametrize('butcher_tableau', (LobattoIIIC(3),
                                             RadauIIA(2)))
def test_pc_acc(butcher_tableau):
    assert heat(butcher_tableau) < 1.e-6
