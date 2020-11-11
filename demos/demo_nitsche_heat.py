from firedrake import *  # noqa: F403

from ufl.algorithms.ad import expand_derivatives

from irksome import LobattoIIIC, Dt, TimeStepper

ButcherTableau = LobattoIIIC(2)
ns = ButcherTableau.num_stages
N = 64

msh = UnitSquareMesh(N, N)
V = FunctionSpace(msh, "CG", 1)
x, y = SpatialCoordinate(msh)

dt = Constant(10.0 / N)
t = Constant(0.0)

# This will give the exact solution at any time t.  We just have
# to t.assign(time_we_want)
uexact = exp(-t) * cos(pi * x) * sin(pi * y)

# MMS works on symbolic differentiation of true solution, not weak form
rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

u = interpolate(uexact, V)

# define the variational form once outside the loop
h = CellSize(msh)
n = FacetNormal(msh)
beta = Constant(100.0)
v = TestFunction(V)
F = (inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v) * dx
     - inner(dot(grad(u), n), v) * ds
     - inner(dot(grad(v), n), u - uexact) * ds
     + beta/h * inner(u - uexact, v) * ds)

params = {"mat_type": "aij",
          "snes_type": "ksponly",
          "ksp_type": "preonly",
          "pc_type": "lu"}

stepper = TimeStepper(F, ButcherTableau, t, dt, u, solver_parameters=params)

while (float(t) < 1.0):
    if (float(t) + float(dt) > 1.0):
        dt.assign(1.0 - float(t))

    stepper.advance()

    t.assign(float(t) + float(dt))
    print(float(t))

print()
print(errornorm(uexact, u)/norm(uexact))
