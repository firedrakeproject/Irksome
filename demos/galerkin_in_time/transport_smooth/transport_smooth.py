from firedrake import *
from irksome import Dt
from fetsome import *

def solve_transport_smooth(Ns, dt, tmax, kt, generator_type, info=False):
    # Testing a linear transport equation on a periodic boundary
    # Explores numerical dissipation of the FET scheme and implementation
    # correctness.
    # The posed problem is:
    #           du/dt + c*du/dx = 0
    # On the interval 0 <= x <= 10 with c = 2.5. The initial condition is:
    #           u(x, 0) = 5*sin^2(pi/10*x)
    # The analytical solution is:
    #           u(x, t) = 5*sin^2(pi/10*(x - ct))

    # Discretisation parameters
    c = as_vector([2.5])
    L = 10

    # Function space parameters
    mesh = PeriodicIntervalMesh(Ns, L)
    V = FunctionSpace(mesh, "CG", 2)

    T = TimeFunctionSpace(kt)
    t = Constant(0.)

    u = TrialFunction(V)
    v = TestFunction(V)
    b = (Dt(u)*v + dot(c, grad(u))*v) * dx

    x = SpatialCoordinate(mesh)[0]
    u0 = interpolate(5*(sin(pi/10*x)**2), V)

    F = (b, None, None)

    timestepper = VariationalTimeStepper(F, u0, u, v, T, t, dt, family=generator_type)

    steps = int(tmax / dt)
    us = [u0]
    for _ in range(steps):
        element_us = timestepper.advance(info=info, include_substages=True)
        us += element_us[1:]

    return us
