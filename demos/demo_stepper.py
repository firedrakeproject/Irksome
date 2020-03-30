from firedrake import *  # noqa: F403

from ufl.algorithms.ad import expand_derivatives

from IRKsome import GaussLegendre, getForm, Dt, TimeStepper

butcher_tableau = GaussLegendre(1)
ns = len(butcher_tableau.b)
N = 100

# Single point of entry in case you want to change the size of the box.
x0 = 0.0
x1 = 10.0
y0 = 0.0
y1 = 10.0

msh = RectangleMesh(N, N, x1, y1)
V = FunctionSpace(msh, "CG", 1)
x, y = SpatialCoordinate(msh)

# Stores initial condition
S = Constant(2.0)
C = Constant(1000.0)
tc = 0
dtc = 10. / N
dt = Constant(dtc)
t = Constant(0.0)


# expressions used in defining the true solution

B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
R = (x * x + y * y) ** 0.5
uexact = B * atan(t)*(pi / 2.0 - atan(S * (R - t)))

# method of manufactured solution
rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

# Holds the initial condition.  This gets overwritten at each time step.
u = interpolate(uexact, V)

# define the variational form once and for all.
v = TestFunction(V)
F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx
bc = DirichletBC(V, 0, "on_boundary")


# Use very simple parameters.  More sophisticated options are possible.
luparams = {"mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu"}

# the TimeStepper object builds the UFL for the multi-stage RK method
# and sets up a solver.  It also provides a method for updating the solution
# at each time step.
# It overwites the UFL constant t with the current time value at each
# step.  u is also updated in place.

stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                      solver_parameters=luparams)

while (float(t) < 1.0):
    stepper.advance()
    print(float(t))
    t.assign(float(t) + float(dt))
 
print()
print(norm(u-uexact)/norm(uexact))
