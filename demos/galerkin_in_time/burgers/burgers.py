from firedrake import *
from irksome import Dt
from fetsome import *

def solve_burgers(Ns, dt, tmax, kt, generator_type, info=False):
    mesh = PeriodicIntervalMesh(Ns, 5)
    V = FunctionSpace(mesh, "CG", 1)
    T = TimeFunctionSpace(kt)
    t = Constant(0.)

    u = TrialFunction(V)
    v = TestFunction(V)

    x = SpatialCoordinate(mesh)[0]
    u0 = interpolate(sin(4*pi/5 * x), V)

    b = (-u*Dt(v) - 1./2.*(u**2)*v.dx(0))*dx
    db = u*v*dx

    F = (b, db, None)
    timestepper = VariationalTimeStepper(F, u0, u, v, T, t, dt, generator_type)

    steps = int(tmax/dt)
    us = [u0]
    for _ in range(steps):
        element_us = timestepper.advance(info=info, include_substages=True)
        us += element_us[1:]
    return us