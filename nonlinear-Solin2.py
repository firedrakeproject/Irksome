from firedrake import *

import numpy
from ufl import replace
from ufl.algorithms.ad import expand_derivatives
import butcher
from nonlinear import getForm

import matplotlib.pyplot as plt


BT = butcher.BackwardEuler()
ns = len(BT.b)
N = 16

# Single point of entry in case you want to change the size of the box.
x0 = 0.0
x1 = 10.0
y0 = 0.0
y1 = 10.0

msh= RectangleMesh(N, N, x1, y1)
V = FunctionSpace(msh, "CG", 1)
x, y = SpatialCoordinate(msh)

# Stores initial condition!
S= Constant(2.0)
C= Constant(1000.0)
dt = Constant(0.1)
t = Constant(0.0)

# We can just let these and the true solutions be expressions!
B= (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
R= (x*x+y*y)**0.5

# This will give the exact solution at any time t.  We just have
# to t.assign(time_we_want)
uexact = B*atan(t)*(pi/2.0-atan(S*(R-t)))    

# MMS works on symbolic differentiation of true solution, not weak form
# Except we might need to futz with this since replacement is breaking on this!
rhs = expand_derivatives(diff(uexact,t) - div(grad(uexact)))

tc=0
dtc=0.1

u = interpolate(uexact, V)

# Build it once and update it rather than deepcopying in the
# time stepping loop.
unew = Function(V)

# define the variational form once outside the loop
# notice that there is no time derivative term.  Our function
# supplies that.
v = TestFunction(V)
F = inner(grad(u), grad(v))*dx - inner(rhs, v)*dx

# hand off the nonlinear function F to get weak form for RK method
Fnew, k = getForm(F, BT, t, dt, u)

# We only need to set up the solver one time!
params = {"mat_type": "aij",
          "ksp_type": "preonly",
          "pc_type": "lu"}

prob = NonlinearVariationalProblem(Fnew, k)
solver = NonlinearVariationalSolver(prob, solver_parameters=params)

# get a tuple of the stages, each as a Coefficient
ks = k.split()

while (tc<5.0):
    solver.solve()

    # update unew
    for i in range(ns):
        unew += dtc * BT.b[i] * ks[i]

    u.assign(unew)

    tc+=dtc
    t.assign(tc)  # takes a new value, not a Constant
    print(tc, assemble(u**2*dx))

plot(u)
plt.show()
