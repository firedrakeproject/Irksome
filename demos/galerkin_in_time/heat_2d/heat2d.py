from firedrake import *
from irksome import Dt
from irksome.fetsome import TimeFunctionSpace, VariationalTimeStepper

def solve_heat_2d_forced(Ns, dt, tmax, kt, generator_type, info=False):
    # Heat equation to be solved in 2 dimensions comes from the desired
    # solution:
    #       u(x,y,t) = pi * sin^2(pi*x) * sin^2(pi*y) * e^(-1/2 t)
    # This function satisfies the vanishing Neumann boundary conditions on a
    # rectangle for any integer length sides. The initial condition would
    # therefore be:
    #       u_0(x,y) = pi * sin^2(pi*x) * sin^2(pi*y)
    # Using the method of manufactured solutions we obtain the forcing
    # function:
    #       f(x,y,t) = -(pi/2*sin^2(pi*x)*sin^2(pi*y)
    #                    + 2pi^3*cos(2pi*x)*sin^2(pi*y)
    #                    + 2pi^3*sin^2(pi*x)*cos(2pi*y))*e^(-1/2)t

    # Discretisation parameters
    L = 2.
    steps = int(tmax / dt)

    # Space and time definitions
    mesh = SquareMesh(Ns, Ns, L)
    V = FunctionSpace(mesh, "CG", 2)
    T = TimeFunctionSpace(kt)

    # Define trial and test functions, variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    b = (Dt(u)*v + dot(grad(u), grad(v)))*dx

    # Define initial condition and forcing function
    x, y = SpatialCoordinate(mesh)
    t = Constant(0)
    u0 = interpolate(pi*(sin(pi*x)**2)*(sin(pi*y)**2), V)
    L = (-(pi/2)*(sin(pi*x)**2)*(sin(pi*y)**2)
         - 2*pi**3*cos(2*pi*x)*(sin(pi*y)**2)
         - 2*pi**3*(sin(pi*x)**2)*cos(2*pi*y))*exp(-1/2*t)*v*dx

    F = (b, None, L)

    stepper = VariationalTimeStepper(F, u0, u, v, T, t, dt, generator_type)

    us = [u0]
    for _ in range(steps):
        element_us = stepper.advance(info, True)
        us += element_us[1:]
    
    return us