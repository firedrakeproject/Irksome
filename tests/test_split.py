from firedrake import (
    UnitIntervalMesh, FunctionSpace, Constant, TestFunction, DirichletBC,
    inner, grad, dx, SpatialCoordinate, sin, pi, interpolate, errornorm)
from irksome import Dt, RadauIIA, TimeStepper, RadauIIAIMEXMethod


def test_split():
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

    bcs = [DirichletBC(V, Constant(2), "on_boundary")]

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
