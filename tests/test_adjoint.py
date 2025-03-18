from firedrake import *
from firedrake.adjoint import *
from irksome import Dt, RadauIIA, TimeStepper


def test_adjoint_diffusivity():
    # continue_annotation()
    msh = UnitSquareMesh(2, 2)
    x, y = SpatialCoordinate(msh)
    V = FunctionSpace(msh, "CG", 1)

    R = FunctionSpace(msh, "R", 0)
    kappa = Function(R, val=2.0)
    # c = Control(kappa)

    v = TestFunction(V)
    u = Function(V)
    u.interpolate(sin(pi*x) * sin(pi * y))

    dt = Constant(0.5)
    t = Constant(0)
    bt = RadauIIA(1)

    bcs = DirichletBC(V, 0, "on_boundary")
    F = inner(Dt(u), v) * dx + kappa * inner(grad(u), grad(v)) * dx
    stepper = TimeStepper(F, bt, t, dt, u, bcs=bcs)

    for _ in range(3):
        stepper.advance()

    J = assemble(inner(u, u) * dx)
    # rf = ReducedFunctional(J, c)
    # assert taylor_test(rf, kappa, Constant(0.1)) > 1.9


test_adjoint_diffusivity()
    
