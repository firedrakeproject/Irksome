import numpy as np
import pytest
from firedrake import (DirichletBC, EquationBC, FunctionSpace, SpatialCoordinate,
                       TestFunction, UnitIntervalMesh, cos, derivative, ds, dx, errornorm, grad,
                       inner, ln, pi, project)
from irksome import Dt, GaussLegendre, MeshConstant, MRIGARK, TimeStepper
from irksome.labeling import explicit
from irksome.tableaux.ButcherTableaux import ButcherTableau

lu = {"snes_type": "ksponly", "ksp_type": "preonly", "pc_type": "lu"}
methods = ["ERK22a", "ERK33a", "ERK45a"]


@pytest.mark.parametrize("name", methods)
def test_slow_tableau(name):
    method = MRIGARK(name)
    assert method.is_explicit
    assert np.isclose(sum(method.b), 1.0)
    assert np.allclose(method.A.sum(axis=1), method.c)
    if method.order >= 2:
        assert np.isclose(method.b @ method.c, 1/2)
    if method.order >= 3:
        assert np.isclose(method.b @ method.c**2, 1/3)
        assert np.isclose(method.b @ method.A @ method.c, 1/6)


def heat(N, degree=2, dirichlet=True):
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", degree)
    x, = SpatialCoordinate(msh)
    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(0.0)
    uexact = cos(t) * (1 + x - x**2) if dirichlet else cos(t) * cos(pi * x)
    rhs = Dt(uexact) - uexact.dx(0).dx(0) + uexact
    u = project(uexact, V)
    v = TestFunction(V)
    Ffast = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
    Fslow = inner(u, v) * dx - inner(rhs, v) * dx
    bc = DirichletBC(V, uexact, "on_boundary")
    return t, dt, u, uexact, Ffast, Fslow, bc


@pytest.mark.parametrize("name", methods)
def test_no_fast_terms(name):
    method = MRIGARK(name)
    sols = []
    for kind in ("mri", "rk"):
        t, dt, u, uexact, Ffast, Fslow, bc = heat(8)
        v = TestFunction(u.function_space())
        F = inner(Dt(u), v) * dx + explicit(Fslow)
        dt.assign(0.01)
        if kind == "mri":
            stepper = TimeStepper(F, method, t, dt, u,
                                  fast_parameters={"tableau": GaussLegendre(2),
                                                   "stepper_kwargs": {"solver_parameters": lu}})
        else:
            slow = ButcherTableau(method.A, method.b, None, method.c, method.order, None, None)
            stepper = TimeStepper(inner(Dt(u), v) * dx + Fslow, slow, t, dt, u,
                                  stage_type="explicit", solver_parameters=lu)
        for _ in range(5):
            stepper.advance()
            t.assign(t + dt)
        sols.append(u.dat.data_ro.copy())
    assert np.allclose(sols[0], sols[1], rtol=0, atol=1e-11)


def convergence(name, Ns, degree, dirichlet):
    method = MRIGARK(name)
    errs = []
    for N in Ns:
        t, dt, u, uexact, Ffast, Fslow, bc = heat(32, degree=degree, dirichlet=dirichlet)
        dt.assign(1 / N)
        stepper = TimeStepper(Ffast + explicit(Fslow), method, t, dt, u, bcs=bc if dirichlet else None,
                              fast_parameters={"tableau": GaussLegendre(2), "num_substeps": 2,
                                               "stepper_kwargs": {"solver_parameters": lu}})
        for _ in range(N):
            stepper.advance()
            t.assign(t + dt)
        errs.append(errornorm(project(uexact, u.function_space()), u))
    return np.log2(errs[0] / errs[1])


@pytest.mark.parametrize("name", methods)
def test_convergence(name):
    assert convergence(name, (4, 8), 4, False) > MRIGARK(name).order - 0.4


@pytest.mark.parametrize("name", methods)
def test_convergence_dirichlet(name):
    assert convergence(name, (8, 16), 2, True) > 1.8


def test_unsupported_options_raise():
    t, dt, u, uexact, Ffast, Fslow, bc = heat(4)
    v = TestFunction(u.function_space())
    F = Ffast + explicit(Fslow)
    fast = {"tableau": GaussLegendre(2)}
    with pytest.raises(NotImplementedError):
        TimeStepper(F, MRIGARK("ERK22a"), t, dt, u, fast_parameters=fast,
                    J=derivative(F.form, u))
    with pytest.raises(NotImplementedError):
        TimeStepper(F, MRIGARK("ERK22a"), t, dt, u, fast_parameters=fast,
                    bcs=EquationBC(inner(u - 1, v) * ds(1) == 0, u, 1))


def test_slow_terms_not_defined_at_zero():
    t, dt, u, uexact, Ffast, Fslow, bc = heat(8)
    v = TestFunction(u.function_space())
    t.assign(1.0)
    dt.assign(0.1)
    F = Ffast + explicit(inner(ln(2 + u) + 1 / t, v) * dx)
    stepper = TimeStepper(F, MRIGARK("ERK33a"), t, dt, u,
                          fast_parameters={"tableau": GaussLegendre(2),
                                           "stepper_kwargs": {"solver_parameters": lu}})
    stepper.advance()
    assert np.isfinite(u.dat.data_ro).all()
