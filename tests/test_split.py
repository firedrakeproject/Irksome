import petsc4py.PETSc
petsc4py.PETSc.Sys.popErrorHandler() # noqa

from firedrake import (
    UnitIntervalMesh, UnitSquareMesh, FunctionSpace, Constant,
    TestFunction, Function, DirichletBC, VectorElement,
    FiniteElement, MixedElement, VectorSpaceBasis, SpatialCoordinate,
    inner, grad, dx, sin, pi, interpolate, split,
    errornorm, as_vector, dot, div, diff)
from irksome import Dt, RadauIIA, TimeStepper, RadauIIAIMEXMethod
from irksome.tools import AI, IA
import pytest
from ufl.algorithms import expand_derivatives


@pytest.mark.parametrize('splitting', (AI, IA))
def test_diffreact(splitting):
    N = 32
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


def NavierStokesTest(N, butcher_tableau, stage_type="deriv", splitting=AI):
    mesh = UnitSquareMesh(N, N)

    Ve = VectorElement("CG", mesh.ufl_cell(), 2)
    Pe = FiniteElement("CG", mesh.ufl_cell(), 1)
    Ze = MixedElement([Ve, Pe])
    Z = FunctionSpace(mesh, Ze)

    t = Constant(0.0)
    dt = Constant(1.0/N)
    (x, y) = SpatialCoordinate(mesh)

    uexact = as_vector([x*t + y**2, -y*t+t*(x**2)])
    pexact = Constant(0, domain=mesh)

    u_rhs = (
        expand_derivatives(diff(uexact, t))
        - div(grad(uexact))
        + grad(pexact)
        + dot(uexact, grad(uexact)))
    p_rhs = -div(uexact)

    z = Function(Z)
    test_z = TestFunction(Z)
    (u, p) = split(z)
    (v, q) = split(test_z)
    F = (inner(Dt(u), v)*dx
         + inner(grad(u), grad(v))*dx
         + inner(dot(u, grad(u)), v)*dx
         - inner(p, div(v))*dx
         - inner(q, div(u))*dx
         - inner(u_rhs, v)*dx
         - inner(p_rhs, q)*dx)

    bcs = [DirichletBC(Z.sub(0), uexact, "on_boundary")]
    nsp = [(1, VectorSpaceBasis(constant=True))]

    u, p = z.split()
    u.interpolate(uexact)

    lu = {"mat_type": "aij",
          "snes_type": "newtonls",
          "snes_linesearch_type": "l2",
          "snes_linesearch_monitor": None,
          "snes_monitor": None,
          "snes_rtol": 1e-8,
          "snes_atol": 1e-8,
          "snes_force_iteration": 1,
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"}

    stepper = TimeStepper(F, butcher_tableau, t, dt, z,
                          stage_type=stage_type,
                          bcs=bcs, solver_parameters=lu, nullspace=nsp)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    (u, p) = z.split()
    print(errornorm(uexact, u))
    print(errornorm(pexact, p))
    return errornorm(uexact, u) + errornorm(pexact, p)


@pytest.mark.parametrize('N', [2**j for j in range(2, 4)])
def test_NavierStokes(N):
    error = NavierStokesTest(N, RadauIIA(2), stage_type="value",
                             splitting=AI)
    print(abs(error))
    assert abs(error) < 4e-8


if __name__ == "__main__":
    test_NavierStokes(4)
