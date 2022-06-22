from firedrake import *
from irksome import Dt
from irksome.fetsome import TimeFunctionSpace, VariationalTimeStepper

def solve_heat_3d_forced(Ns, dt, tmax, kt, generator_type, info=False):
    # Discretisation parameters
    L = 2.
    steps = int(tmax / dt)

    # Space and time definitions
    mesh = CubeMesh(Ns, Ns, Ns, L)
    V = FunctionSpace(mesh, "CG", 2)
    T = TimeFunctionSpace(kt)

    # Define trial and test functions, variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    b = (-u*Dt(v) + dot(grad(u), grad(v)))*dx
    db = u*v*dx

    # Define initial condition and forcing function
    x, y, z = SpatialCoordinate(mesh)
    t = Constant(0)
    u0 = interpolate(pi*(sin(pi*x)**2)*(sin(pi*y)**2)*(sin(pi*z)**2), V)
    L = (-(pi/2)*(sin(pi*x)**2)*(sin(pi*y)**2)*(sin(pi*z)**2)
         - 2*pi**3*cos(2*pi*x)*(sin(pi*y)**2)*(sin(pi*z)**2)
         - 2*pi**3*(sin(pi*x)**2)*cos(2*pi*y)*(sin(pi*z)**2)
         - 2*pi**3*(sin(pi*x)**2)*(sin(pi*y)**2)*cos(2*pi*z))*exp(-1/2*t)*v*dx

    F = (b, db, L)
    bcs = [DirichletBC(V, Constant(0.), "on_boundary")]
    stepper = VariationalTimeStepper(F, u0, u, v, T, t, dt, generator_type, bcs=bcs)

    us = [u0]
    for _ in range(steps):
        element_us = stepper.advance(info, True)
        us += element_us[1:]
    
    return us