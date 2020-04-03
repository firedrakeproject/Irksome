from firedrake import *  # noqa: F403

from ufl.algorithms.ad import expand_derivatives

from IRKsome import LobattoIIIC, getForm, Dt

import matplotlib.pyplot as plt

BT = LobattoIIIC(2)
ns = len(BT.b)
N = 32

# Single point of entry in case you want to change the size of the box.
x0 = 0.0
x1 = 10.0
y0 = 0.0
y1 = 10.0
msh = RectangleMesh(N, N, x1, y1)
V = FunctionSpace(msh, "RT", 2)
W = FunctionSpace(msh, "DG", 1)
Z = V * W
x, y = SpatialCoordinate(msh)

# Stores initial condition!
S = Constant(2.0)
C = Constant(1000.0)

dt = Constant(10.0 / N)
t = Constant(0.0)

# We can just let these and the true solutions be expressions
B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
R = (x * x + y * y) ** 0.5
pexact = B * atan(t)*(pi / 2.0 - atan(S * (R - t)))

# MMS works on symbolic differentiation of true solution, not weak form
rhs = expand_derivatives(diff(pexact, t)) - div(grad(pexact))

# Holds the initial condition
up = project(as_vector([0, 0, pexact]), Z)
u, p = split(up)

upnew = project(up, Z)

# define the variational form once outside the loop
v, w = TestFunctions(Z)

F = (inner(Dt(p), w) * dx + inner(div(u), w) * dx - inner(rhs, w) * dx
     + inner(u, v) * dx - inner(p, div(v)) * dx)

Fnew, k, _, _ = getForm(F, BT, t, dt, up)

# We only need to set up the solver one time!
params = {"mat_type": "aij",
          "ksp_type": "preonly",
          "pc_type": "lu"}

prob = NonlinearVariationalProblem(Fnew, k)
solver = NonlinearVariationalSolver(prob, solver_parameters=params)


# get a tuple of the stages, each as a Coefficient
ks = k.split()
upnewbits = upnew.split()
num_fields = len(upnewbits)

while (float(t) < 1.0):
    if (float(t) + float(dt) > 1.0):
        dt.assign(1.0 - float(t))

    solver.solve()

    # update
    for i in range(ns):
        for j in range(num_fields):
            up.dat.data[j][:] += float(dt) * BT.b[i] * ks[i*num_fields+j].dat.data[:]

    t.assign(float(t) + float(dt))
    print(float(t))

u, p = up.split()
print(errornorm(pexact, p) / norm(pexact))
