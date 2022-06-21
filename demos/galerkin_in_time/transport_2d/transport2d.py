from firedrake import *
from irksome import Dt
from fetsome import *

def solve_transport_2d(Ns, dt, tmax, kt, generator_type, info=False):
    # Testing a linear transport equation on a periodic-in-x domain.
    # The posed problem is:
    #           du/dt + c.grad u = 0
    # On the interval 0 <= x <= 10 with c = [2.5, 0]. The initial condition is:
    #           u(x, 0) = 5*sin^2(pi/10*x)*sin(pi/5*y)
    # The analytical solution is:
    #           u(x, t) = 5*sin^2(pi/10*(x - ct))*sin(pi/5*y)

    # Discretisation parameters
    c = as_vector([2.5, 0])
    Lx = 10
    Ly = 5

    # Function space parameters
    mesh = PeriodicRectangleMesh(Ns, Ns // 2, Lx, Ly, direction="x")
    V = FunctionSpace(mesh, "CG", 3)

    T = TimeFunctionSpace(kt)
    t = Constant(0.)

    u = TrialFunction(V)
    v = TestFunction(V)
    b = (u*Dt(v) + u*dot(c, grad(v))) * dx
    db = -u*v*dx

    x, y = SpatialCoordinate(mesh)
    u0 = interpolate(5*(sin(pi/10*x)**2)*sin(pi/5*y), V)

    F = (b, db, None)

    timestepper = VariationalTimeStepper(F, u0, u, v, T, t, dt, family=generator_type)

    steps = int(tmax / dt)
    us = [u0]
    for _ in range(steps):
        element_us = timestepper.advance(info=info, include_substages=True)
        us += element_us[1:]

    return us