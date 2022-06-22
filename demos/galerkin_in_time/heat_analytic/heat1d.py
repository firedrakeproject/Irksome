from firedrake import *
from irksome import Dt
from irksome.fetsome import TimeFunctionSpace, VariationalTimeStepper

def solve_heat_analytic(Ns, dt, tmax, kt, generator_type, info=False):
    # Test solution for the heat equation with Neumann boundary conditions in 1-dimension
    # using the problem posed on 0 < x < L, t > 0 (t <= 10, added) with the initial condition:
    #               u(x, 0) = f(x) = 1/2 * cos(2pi/L * x)
    # The analytical solution (from Andrew Walton, Multivariable Calculus 2019-20 ICL) is:
    #               u(x, t) = 1/2 * cos(2pi/L * x) * exp(-4pi^2/L^2 * t)


    # Spatial discretisation parameters (length of problem domain)
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
    b = (Dt(u)*v + dot(grad(u), grad(v))) * dx

    # Compose the initial condition: f(x) = 1/2 * cos(2pi/L * x)
    x = SpatialCoordinate(mesh)[0]
    u0 = interpolate((1./2.)*cos((2*pi/L)*x), V)

    # Compose the FET form triple
    F = (b, None, None)

    # Initialise the timestepper
    timestepper = VariationalTimeStepper(F, u0, u, v, T, t, dt, family=generator_type)
    
    us = [u0]
    for _ in range(steps):
        element_us = timestepper.advance(info, include_substages=True)
        us += element_us[1:]
    
    return us