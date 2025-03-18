from firedrake import *
from firedrake.adjoint import *
from irksome import Dt, RadauIIA, TimeStepper


def test_adjoint_diffusivity():
    pause_annotation()
    msh = UnitIntervalMesh(8)
    x, = SpatialCoordinate(msh)
    V = FunctionSpace(msh, "CG", 1)
    
    R = FunctionSpace(msh, "R", 0)
    kappa = Function(R).assign(2.0)
    
    v = TestFunction(V)
    u = Function(V)
    u.interpolate(sin(pi*x))
    
    dt = Constant(0.1)
    t = Constant(0)
    nt = 2
    bt = RadauIIA(1)

    params = {
        'snes_type': 'ksponly',
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }
    
    bcs = DirichletBC(V, 0, "on_boundary")
    F = inner(Dt(u), v) * dx + kappa * inner(grad(u), grad(v)) * dx
    stepper = TimeStepper(F, bt, t, dt, u, bcs=bcs,
                          solver_parameters=params)
    
    continue_annotation()
    with set_working_tape() as tape:
        for _ in range(nt):
            stepper.advance()
            # stepper.solver.solve()
            # stepper._update()
            # stepper.u0 += dt*stepper.stages
        J = assemble(inner(u, u) * dx)
        rf = ReducedFunctional(J, Control(kappa), tape=tape)
    pause_annotation()
    
    nu = Function(R).assign(3.0)
    
    assert abs(rf(kappa) - rf(nu)) > 1e-8
    assert taylor_test(rf, kappa, Constant(0.01)) > 1.9

if __name__ == '__main__':
    test_adjoint_diffusivity()
