from firedrake import *
from irksome import Dt
from irksome.fetsome import *

def solve_heat_tdg(Ns, dt, tmax, kt, generator_type, info=False):
    L = 7.

    # Temporal discretisation parameters (steps, number of nodes)
    # Interval and number of steps to cover it
    steps = int(tmax / dt)

    # Define the spatial mesh and the temporal solution function space
    mesh = IntervalMesh(Ns, L)
    V = FunctionSpace(mesh, "CG", 2)    

    # Define the time function space and time variable
    T = TimeFunctionSpace(kt)
    t = Constant(0.)

    # Define the spacetime problem
    u = TrialFunction(V)
    v = TestFunction(V)

    # Compose the initial condition: f(x) = 1/2 * cos(2pi/L * x)
    x = SpatialCoordinate(mesh)[0]
    u0 = interpolate((1./2.)*cos((2*pi/L)*x), V)
    
    b = (-u*Dt(v) + dot(grad(u), grad(v))) * dx
    db = u*v*dx

    # Compose the FET form triple
    F = (b, db, None)

    # Initialise the timestepper
    timestepper = VariationalTimeStepper(F, u0, u, v, T, t, dt, family=generator_type)
    
    us = [u0]
    for _ in range(steps):
        element_us = timestepper.advance(info, include_substages=True)
        us += element_us[1:]
    
    return us