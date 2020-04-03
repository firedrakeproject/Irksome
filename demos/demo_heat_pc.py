# Demonstrate a block preconditioner for multi-stage RK method.
# This is in the spirit suggested by Mardal/Nilssen/Staff --
# use the block diagonal of the operator and hit those blocks with
# algebraic multigrid

from firedrake import *  # noqa: F403

from ufl.algorithms.ad import expand_derivatives

from IRKsome import LobattoIIIC, getForm, Dt

ButcherTableau = LobattoIIIC(2)

ns = ButcherTableau.num_stages

N = 64

# Single point of entry in case you want to change the size of the box.
x0 = 0.0
x1 = 10.0
y0 = 0.0
y1 = 10.0

msh = RectangleMesh(N, N, x1, y1)
V = FunctionSpace(msh, "CG", 1)
x, y = SpatialCoordinate(msh)

# Stores initial condition!
S = Constant(2.0)
C = Constant(1000.0)
dt = Constant(10. / N)
t = Constant(0.0)

# We can just let these and the true solutions be expressions!
B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
R = (x * x + y * y) ** 0.5

# This will give the exact solution at any time t.  We just have
# to t.assign(time_we_want)
uexact = B * atan(t)*(pi / 2.0 - atan(S * (R - t)))

# MMS works on symbolic differentiation of true solution, not weak form
# Except we might need to futz with this since replacement is breaking on this!
rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

u = interpolate(uexact, V)

# define the variational form once outside the loop
v = TestFunction(V)
F =  inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx

bc = DirichletBC(V, 0, "on_boundary")

# hand off the nonlinear function F to get weak form for RK method
Fnew, k, bcnew, bcdata = getForm(F, ButcherTableau, t, dt, u, bcs=bc)

params = {"mat_type": "matfree",
          "snes_type": "ksponly",
          "ksp_type": "gmres",
          "ksp_monitor": None,
          "pc_type": "fieldsplit",   # block preconditioner
          "pc_fieldsplit_type": "additive" # block diagaonal
}

# on each block, apply a sweep of gamg (algebraic multigrid) with default
# parameters.

per_field = {"ksp_type": "preonly",
             "pc_type": "python",
             "pc_python_type": "firedrake.AssembledPC",
             "assembled_pc_type": "gamg"}

for s in range(ns):
    params["fieldsplit_%s" % (s,)] = per_field

prob = NonlinearVariationalProblem(Fnew, k, bcs=bcnew)
solver = NonlinearVariationalSolver(prob, solver_parameters=params)

solver.solve()
