import petsc4py.PETSc
petsc4py.PETSc.Sys.popErrorHandler() # noqa

from firedrake import (
    UnitIntervalMesh, UnitSquareMesh, FunctionSpace, Constant,
    TestFunction, Function, DirichletBC, VectorElement,
    FiniteElement, MixedElement, VectorSpaceBasis, SpatialCoordinate,
    assemble, inner, grad, dx, sin, pi, interpolate, split,
    errornorm, as_vector, dot, div)
from irksome import Dt, RadauIIA, TimeStepper, RadauIIAIMEXMethod
from irksome.tools import AI, IA
import pytest


@pytest.mark.parametrize('splitting', (AI, IA))
def test_diffreact(splitting):
    N = 16
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", 2)
    v = TestFunction(V)

    k = 2
    bt = RadauIIA(k)

    dt = Constant(1.0 / N)
    t = Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    uIC = 2 + sin(2 * pi * x)

    u_imp = interpolate(uIC, V)
    u_split = interpolate(uIC, V)

    bcs = DirichletBC(V, Constant(2), "on_boundary")

    Ffull = (
        inner(Dt(u_imp), v) * dx + inner(grad(u_imp), grad(v)) * dx
        - inner(u_imp * (1-u_imp), v) * dx)

    Fimp = (
        inner(Dt(u_split), v) * dx + inner(grad(u_split), grad(v)) * dx)
    Fexp = -inner(u_split * (1 - u_split), v) * dx

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    rk_stepper = TimeStepper(
        Ffull, bt, t, dt, u_imp, bcs=bcs,
        solver_parameters=luparams,
        stage_type="value")

    imex_stepper = RadauIIAIMEXMethod(
        Fimp, Fexp, bt, t, dt, u_split, bcs=bcs,
        splitting=splitting,
        it_solver_parameters=luparams,
        prop_solver_parameters=luparams)

    num_iter_init = 10
    for i in range(num_iter_init):
        imex_stepper.iterate()

    num_iter_perstep = 5
    t_end = 10 * float(dt)
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        rk_stepper.advance()
        imex_stepper.advance()

        for i in range(num_iter_perstep):
            imex_stepper.iterate()
        t.assign(float(t) + float(dt))

    assert errornorm(u_split, u_imp) < 1.e-10


def NavierStokesSplitTest(N, num_stages):
    mesh = UnitSquareMesh(N, N)
    butcher_tableau = RadauIIA(num_stages)

    Ve = VectorElement("CG", mesh.ufl_cell(), 2)
    Pe = FiniteElement("CG", mesh.ufl_cell(), 1)
    Ze = MixedElement([Ve, Pe])
    Z = FunctionSpace(mesh, Ze)

    t = Constant(0.0)
    dt = Constant(1.0/N)

    z_imp = Function(Z)
    z_split = Function(Z)
    test_z = TestFunction(Z)

    x, y = SpatialCoordinate(mesh)

    def Fimp(z, test):
        u, p = split(z)
        v, q = split(test)
        return (inner(Dt(u), v)*dx
                + inner(grad(u), grad(v))*dx
                - inner(p, div(v))*dx
                - inner(q, div(u))*dx)

    def Fexp(z, test):
        u, _ = split(z)
        v, _ = split(test)
        return inner(dot(u, grad(u)), v) * dx

    def Ffull(z, test):
        return Fimp(z, test) + Fexp(z, test)

    bcs = [DirichletBC(Z.sub(0), as_vector([x*(1-x), 0]), (4,)),
           DirichletBC(Z.sub(0), as_vector([0, 0]), (1, 2, 3))]

    nsp = [(1, VectorSpaceBasis(constant=True))]

    lunl = {
        "mat_type": "aij",
        "snes_type": "newtonls",
        "snes_linesearch_type": "l2",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-10,
        "snes_force_iteration": 1,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"}

    F_full = Ffull(z_imp, test_z)
    F_imp = Fimp(z_split, test_z)
    F_exp = Fexp(z_split, test_z)

    imp_stepper = TimeStepper(
        F_full, butcher_tableau, t, dt, z_imp,
        stage_type="value",
        bcs=bcs, solver_parameters=lunl, nullspace=nsp)

    imex_stepper = RadauIIAIMEXMethod(
        F_imp, F_exp, butcher_tableau, t, dt, z_split,
        bcs=bcs, nullspace=nsp,
        it_solver_parameters=lunl, prop_solver_parameters=lunl)

    num_iter_init = 10
    for i in range(num_iter_init):
        imex_stepper.iterate()

    num_iter_perstep = 5

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        imp_stepper.advance()
        imex_stepper.advance()
        for i in range(num_iter_perstep):
            imex_stepper.iterate()
        t.assign(float(t) + float(dt))
        uimp, pimp = z_imp.split()
        pimp -= assemble(pimp*dx)
        usplit, psplit = z_split.split()
        psplit -= assemble(psplit*dx)
        print(errornorm(uimp, usplit), errornorm(pimp, psplit))

    return errornorm(uimp, usplit) + errornorm(pimp, psplit)


@pytest.mark.parametrize('N', [2**j for j in range(3, 5)])
@pytest.mark.parametrize('num_stages', [1, 2])
def test_SplitNavierStokes(N, num_stages):
    error = NavierStokesSplitTest(N, num_stages)
    print(abs(error))
    assert abs(error) < 4e-8


if __name__ == "__main__":
    NavierStokesSplitTest(4, 2)
