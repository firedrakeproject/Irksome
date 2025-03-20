import numpy
import pytest
from firedrake import (DirichletBC, Function, FunctionSpace, SpatialCoordinate,
                       TestFunction, UnitSquareMesh, div, dx, errornorm, grad, inner)
from irksome import (Dt, NystromAuxiliaryOperatorPC,
                     StageDerivativeNystromTimeStepper, LobattoIIIC, MeshConstant,
                     GaussLegendre)


def Fubc(V, t, uexact):
    u = Function(V)
    u.interpolate(uexact)
    v = TestFunction(V)
    rhs = Dt(uexact, 2) - div(grad(uexact)) - uexact * (1-uexact)
    F = inner(Dt(u, 2), v)*dx + inner(grad(u), grad(v))*dx - inner(u*(1-u), v)*dx - inner(rhs, v)*dx

    bc = DirichletBC(V, uexact, "on_boundary")

    return (F, u, Function(V), bc)


class myPC(NystromAuxiliaryOperatorPC):
    def getNewForm(self, pc, u0, ut0, test):
        appctx = self.get_appctx(pc)
        bcs = appctx["bcs"]
        F = inner(Dt(u0, 2), test) * dx + inner(grad(u0), grad(test)) * dx
        return F, bcs


def rd(butcher_tableau):
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

    luparams = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu"}

    clinesLD = {
        "mat_type": "aij",
        "ksp_type": "gmres",
        "ksp_monitor": None,
        "pc_type": "python",
        "pc_python_type": "irksome.ClinesLD",
        "aux": {
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "multiplicative",
            "fieldsplit": {
                "ksp_type": "preonly",
                "pc_type": "gamg"
            }
        }
    }

    mypc_params = {
        "mat_type": "matfree",
        "ksp_type": "gmres",
        "pc_type": "python",
        "pc_python_type": "test_pc.myPC",
        "aux": {
            "pc_type": "lu"}
    }

    params = [luparams, clinesLD, mypc_params]

    for solver_parameters in params:
        F, u, t, bc = Fubc(V, t, uexact)

        stepper = StageDerivativeNystromTimeStepper(F, butcher_tableau, t, dt, u, t, bcs=bc,
                                                    solver_parameters=solver_parameters)
        stepper.advance()
        sols.append(u)

    errs = [errornorm(sols[0], uu) for uu in sols[1:]]
    return numpy.max(errs)


@pytest.mark.parametrize('butcher_tableau', (LobattoIIIC(3),
                                             GaussLegendre(2)))
def test_pc_acc(butcher_tableau):
    assert rd(butcher_tableau) < 1.e-6
