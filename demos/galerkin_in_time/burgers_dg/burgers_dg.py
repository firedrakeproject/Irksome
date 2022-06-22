from firedrake import *
from irksome import Dt
from irksome.fetsome import *

def solve_burgers_dg_space(Ns, dt, tmax, kt, generator_type, info=False):
    L = 5.

    mesh = PeriodicIntervalMesh(Ns, L)
    V = FunctionSpace(mesh, "DG", 1)
    T = TimeFunctionSpace(kt)
    t = Constant(0.)

    u = TrialFunction(V)
    v = TestFunction(V)
    n = FacetNormal(mesh)

    x = SpatialCoordinate(mesh)[0]
    u0 = interpolate(sin(4*pi/5 * x), V)
    
    b = (-u*Dt(v) - 1./2.*(u**2)*v.dx(0))*dx \
        + (0.5*(0.5*u('+')**2 + 0.5*u('-')**2) - 0.5*(u('+')*n('-')[0] + u('-')*n('+')[0])) \
        * (n('+')[0]*v('+') + n('-')[0]*v('-'))*dS

    db = u*v*dx

    F = (b, db, None)
    timestepper = VariationalTimeStepper(F, u0, u, v, T, t, dt, family=generator_type)

    steps = int(tmax/dt)
    us = [u0]
    for _ in range(steps):
        element_us = timestepper.advance(info=info, include_substages=True)
        us += element_us[1:]
    return us