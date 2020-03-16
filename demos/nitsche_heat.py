from firedrake import *  # noqa: F403

from ufl.algorithms.ad import expand_derivatives

from IRKsome import GaussLegendre, BackwardEuler, LobattoIIIA, Radau35, Radau23, getForm

BT = Radau23()
ns = len(BT.b)
N = 32

msh = UnitSquareMesh(N, N)
V = FunctionSpace(msh, "CG", 1)
x, y = SpatialCoordinate(msh)

tc = 0
dtc = 1 / N
dt = Constant(dtc)
t = Constant(0.0)

# This will give the exact solution at any time t.  We just have
# to t.assign(time_we_want)
uexact = exp(-t) * cos(pi * x) * sin(pi * y)

# MMS works on symbolic differentiation of true solution, not weak form
rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

u = interpolate(uexact, V)

# define the variational form once outside the loop
# notice that there is no time derivative term.  Our function
# supplies that.
h = CellSize(msh)
n = FacetNormal(msh)
beta = Constant(100.0)
v = TestFunction(V)
F = (inner(grad(u), grad(v))*dx - inner(rhs, v) * dx
     - inner(dot(grad(u), n), v) * ds
     - inner(dot(grad(v), n), u - uexact) * ds
     + beta/h * inner(u - uexact, v) * ds)


# hand off the nonlinear function F to get weak form for RK method
# No DirichletBC objects.
Fnew, k, _, _ = getForm(F, BT, t, dt, u)
fs = Fnew.arguments()[0].function_space()

params={"mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu"}

prob = NonlinearVariationalProblem(Fnew, k)
solver = NonlinearVariationalSolver(prob, solver_parameters=params)

# get a tuple of the stages, each as a Coefficient
ks = k.split()

while (tc < 1.0):
    solver.solve()

    for i in range(ns):
        u += dtc * BT.b[i] * ks[i]

    tc += dtc
    t.assign(tc)  # takes a new value, not a Constant
    print(tc)

print()
print(errornorm(uexact, u)/norm(uexact))
