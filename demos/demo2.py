from ufl import replace

from firedrake import *  # noqa: F403

from ufl.algorithms.ad import expand_derivatives
from ufl import replace

from IRKsome import LobattoIIIA, getForm

BT = LobattoIIIA(2)

ns = len(BT.b)
N = 32
msh = UnitSquareMesh(N, N)

V = FunctionSpace(msh, "CG", 1)
x, y = SpatialCoordinate(msh)
tc = 0
dtc = 1.0 / N
dt = Constant(dtc)
t = Constant(0.0)

uexact = exp(-t) * cos(pi * x) * sin(pi * y)

# MMS works on symbolic differentiation of true solution,
rhs = expand_derivatives(diff(uexact,t) - div(grad(uexact)))

# Initial condition
u = interpolate(uexact, V)

# define the variational form once outside the loop
# notice that there is no time derivative term.  Our function
# supplies that.
v = TestFunction(V)

F = inner(grad(u), grad(v))*dx - inner(rhs, v)*dx
bcs= DirichletBC(V, uexact, "on_boundary") 

Fnew, k, bcnew, gblah = getForm(F, BT, t, dt, u, bcs=bcs)

params = {"mat_type": "aij",
          "ksp_type": "preonly",
          "pc_type": "lu"}

# get a tuple of the stages, each as a Coefficient
ks = k.split()

prob = NonlinearVariationalProblem(Fnew, k, bcs=bcnew)
solver = NonlinearVariationalSolver(prob, solver_parameters=params)


while (tc<1):
    for (gdat, gcur) in gblah:
        gdat.interpolate(gcur)

    solver.solve()

    for i in range(ns):
        u += dtc * BT.b[i] * ks[i]

    #unew.assign(u)

    tc+=dtc
    t.assign(tc)  # takes a new value, not a Constant
    print(tc)

     #real solution evaluated at t=2

#ueval = exp(-t)*cos(pi*x)*sin(pi*y)
print(errornorm(uexact, u))
# plot(interpolate(ueval,V) )
# plot(unew)
