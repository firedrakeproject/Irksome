from firedrake import *
from irksome import Dt
from fetsome import *

def solve_burgers_exact(Ns, dt, tmax, kt, generator_type, info=False):
    L = 8.
    nu = 0.05
    alpha = 1.5
    beta = 1.55
    k = pi / 2.0

    mesh = PeriodicIntervalMesh(Ns, L)
    V = FunctionSpace(mesh, "CG", 3)
    T = TimeFunctionSpace(kt)
    t = Constant(0.)

    u = Function(V)
    v = TestFunction(V)

    x = SpatialCoordinate(mesh)[0]
    u0 = interpolate((2*nu*alpha*k*sin(k*x)) / (beta + alpha*cos(k*x)), V)

    b = (-u*Dt(v) - 1./2.*(u**2)*v.dx(0) + nu*u.dx(0)*v.dx(0))*dx
    
    db = u*v*dx

    F = (b, db, None)
    timestepper = VariationalTimeStepper(F, u0, u, v, T, t, dt, family=generator_type)

    steps = int(tmax/dt)
    us = [u0]
    for _ in range(steps):
        element_us = timestepper.advance(info=info, include_substages=True)
        us += element_us[1:]
    return us