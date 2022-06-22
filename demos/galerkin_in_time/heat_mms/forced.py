from firedrake import *
from irksome import Dt
from irksome.fetsome import TimeFunctionSpace, VariationalTimeStepper

def solve_heat_forced(Ns, dt, tmax, kt, generator_type, info=False):
    # This solves the heat equation using the method of manufactured
    # solutions. The chosen solution is:
    #       u(x, t) = (1/100 x^3 - 3/20 x^2 + 5) e^-t
    # By substitution into the heat equation, this gives the forcing
    # function:
    #       f(x, t) = (-1/100 x^3 + 3/20 x^2 - 3/50 x - 53/10) e^-t

    # Discretisation parameters
    L = 10.
    steps = int(tmax / dt)

    # Spatial mesh, spatial function space and time 
    mesh = IntervalMesh(Ns, L)
    V = FunctionSpace(mesh, "CG", 2)
    T = TimeFunctionSpace(kt)

    # Define the trial and test functions, compose the variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    b = (-u*Dt(v) + dot(grad(u), grad(v))) * dx
    db = u*v*dx

    # Compose the initial condition and the forcing function
    x = SpatialCoordinate(mesh)[0]
    t = Constant(0)
    u0 = interpolate(1/100 * x**3 - 3/20 * x**2 +5, V)
    L = ((-1/100 * x**3 + 3/20 * x**2 - 3/50 * x - 47/10) * exp(-t))*v*dx

    # Compose the form triplet for bilinear, boundary and forcing forms
    F = (b, db, L)

    timestepper = VariationalTimeStepper(F, u0, u, v, T, t, dt, generator_type)

    us = [u0]
    for _ in range(steps):
        element_us = timestepper.advance(info, True)
        us += element_us[1:]
    
    return us