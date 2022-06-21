from firedrake import *
from irksome import Dt
from fetsome import *

def solve_burgers_nu(Ns, dt, tmax, kt, generator_type, info=False):
    mesh = PeriodicIntervalMesh(Ns, 2)
    V = FunctionSpace(mesh, "CG", 2)
    u = TrialFunction(V)
    v = TestFunction(V)

    T = TimeFunctionSpace(kt)
    t = Constant(0.)

    nu = Constant(1e-2)
    b = (Dt(u)*v + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

    x = SpatialCoordinate(mesh)[0]
    u0 = interpolate(sin(2*pi*x), V)

    F = (b, None, None)
    timestepper = VariationalTimeStepper(F, u0, u, v, T, t, dt, generator_type)

    steps = int(tmax/dt)
    us = [u0]
    for _ in range(steps):
        element_us = timestepper.advance(info=info, include_substages=True)
        us += element_us[1:]
    return us