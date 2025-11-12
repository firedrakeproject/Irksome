import numpy
import pytest
from firedrake import (DirichletBC, Function, FunctionSpace, SpatialCoordinate,
                       TestFunction, UnitSquareMesh, div, dx, errornorm,
                       grad, inner)
from irksome import (Dt, GalerkinCollocationScheme, IRKAuxiliaryOperatorPC, LobattoIIIC, MeshConstant,
                     RadauIIA, TimeStepper)

# Tests that various PCs are actually getting the right answer.


def Fubc(V, t, uexact):
    u = Function(V)
    u.interpolate(uexact)
    v = TestFunction(V)
    rhs = Dt(uexact) - div(grad(uexact)) - uexact * (1-uexact)
    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(u*(1-u), v)*dx
    F -= inner(rhs, v)*dx

    bc = DirichletBC(V, uexact, "on_boundary")

    return (F, u, bc)


class myPC(IRKAuxiliaryOperatorPC):
    def getNewForm(self, pc, u0, test):
        appctx = self.get_appctx(pc)
        bcs = appctx["stepper"].orig_bcs
        F = inner(Dt(u0), test) * dx + inner(grad(u0), grad(test)) * dx
        return F, bcs


def rd(scheme):
    N = 4
    msh = UnitSquareMesh(N, N)

    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)

    deg = 2
    V = FunctionSpace(msh, "CG", deg)

    x, y = SpatialCoordinate(msh)

    uexact = t*(x+y)

    sols = []

    luparams = {"mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu"}

    per_field = {"ksp_type": "preonly",
                 "pc_type": "lu"}

    ranaLD = {"mat_type": "aij",
              "ksp_type": "gmres",
              "ksp_converged_reason": None,
              "pc_type": "python",
              "pc_python_type": "irksome.RanaLD",
              "aux": {
                  "pc_type": "fieldsplit",
                  "pc_fieldsplit_type": "multiplicative",
                  "fieldsplit": per_field,
              }}

    ranaDU = {"mat_type": "aij",
              "ksp_type": "gmres",
              "ksp_converged_reason": None,
              "pc_type": "python",
              "pc_python_type": "irksome.RanaDU",
              "aux": {
                  "pc_type": "fieldsplit",
                  "pc_fieldsplit_type": "multiplicative",
                  "fieldsplit": per_field,
              }}

    mypc_params = {"mat_type": "aij",
                   "ksp_type": "gmres",
                   "ksp_converged_reason": None,
                   "pc_type": "python",
                   "pc_python_type": "test_pc.myPC",
                   "aux": {
                       "pc_type": "lu"}}

    params = [luparams, ranaLD, ranaDU, mypc_params]

    for solver_parameters in params:
        F, u, bc = Fubc(V, t, uexact)

        stepper = TimeStepper(F, scheme, t, dt, u, bcs=bc,
                              solver_parameters=solver_parameters)
        stepper.advance()
        sols.append(u)

    errs = [errornorm(sols[0], uu) for uu in sols[1:]]
    return numpy.max(errs)


@pytest.mark.parametrize('butcher_tableau,order', [
    (RadauIIA, 2),
    (LobattoIIIC, 3),
])
def test_pc_acc(butcher_tableau, order):
    assert rd(butcher_tableau(order)) < 1.e-6


@pytest.mark.parametrize("stage_type", ("deriv",))
@pytest.mark.parametrize('quad_scheme,order', [
    ("radau", 2),
    # ("lobatto", 3),
])
def test_rana_galerkin(quad_scheme, order, stage_type):
    scheme = GalerkinCollocationScheme(order, quadrature_scheme=quad_scheme, stage_type=stage_type)
    assert rd(scheme) < 1.e-6
