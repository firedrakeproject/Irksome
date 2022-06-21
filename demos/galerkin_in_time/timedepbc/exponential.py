from firedrake import *
from irksome import Dt
from fetsome import *

def solve_exponential(Ns, dt, tmax, kt, generator_type, info=False):
    # Very simple exponential to solve with time dependent Dirichlet BCs

    mesh = UnitIntervalMesh(Ns)
    V = FunctionSpace(mesh, "CG", 2)
    T = TimeFunctionSpace(kt)
    t = Constant(0.)

    u = TrialFunction(V)
    v = TestFunction(V)

    u0 = interpolate(Constant(1), V)
    b = (Dt(u)*v - 1./2.*u*v)*dx
    F = (b, None, None)

    bcs = [TimeDirichletBC(V, exp(1./2. * t), "on_boundary")]

    timestepper = VariationalTimeStepper(F, u0, u, v, T, t, dt, family=generator_type, bcs=bcs)

    steps = int(tmax / dt)
    us = [u0]
    for _ in range(steps):
        element_us = timestepper.advance(info, include_substages=True)
        us += element_us[1:]
    
    return us