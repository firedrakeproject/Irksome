from firedrake import *  # noqa: F403

from ufl.algorithms.ad import expand_derivatives

from irksome import GaussLegendre, getForm, Dt

BT = GaussLegendre(1)
ns = len(BT.b)
N = 128
coarseN = 8 # size of coarse grid

# Single point of entry in case you want to change the size of the box.
x0 = 0.0
x1 = 10.0
y0 = 0.0
y1 = 10.0

from math import log
nrefs = log(N/coarseN, 2)
assert nrefs == int(nrefs)
nrefs = int(nrefs)
base = RectangleMesh(coarseN, coarseN, x1, y1)
mh = MeshHierarchy(base, nrefs)
msh = mh[-1]
V = FunctionSpace(msh, "CG", 1)
x, y = SpatialCoordinate(msh)

# Stores initial condition!
S = Constant(2.0)
C = Constant(1000.0)
dt = Constant(10. / N)
t = Constant(0.0)

# We can just let these and the true solutions be expressions
B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
R = (x * x + y * y) ** 0.5
uexact = B * atan(t)*(pi / 2.0 - atan(S * (R - t)))

# MMS works on symbolic differentiation of true solution, not weak form
rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

# Holds the initial condition
u = interpolate(uexact, V)

# define the semi-discrete form using Dt from irksome.
v = TestFunction(V)
F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx

bc = DirichletBC(V, 0, "on_boundary")

# hand off the nonlinear function F to get weak form for RK method
Fnew, k, bcnew, bcdata = getForm(F, BT, t, dt, u, bcs=bc)


# monolithic MG for all stages concurrently, using a pointwise block Jacobi
# smoother.

mgparams = {"mat_type": "aij",
            "snes_type": "ksponly",
            #"snes_monitor": None,
            "ksp_type": "fgmres",
            "ksp_monitor_true_residual": None,
            "pc_type": "mg",
            "mg_levels_ksp_type": "chebyshev",
            "mg_levels_ksp_norm_type": "unpreconditioned",
            #"mg_levels_ksp_monitor_true_residual": None,
            "mg_levels_pc_type": "pbjacobi",
            "mg_coarse_pc_type": "lu",
            "mg_coarse_pc_factor_mat_solver_type": "mumps"}


prob = NonlinearVariationalProblem(Fnew, k, bcs=bcnew)
solver = NonlinearVariationalSolver(prob, solver_parameters=mgparams)

# get a tuple of the stages, each as a Coefficient
ks = k.split()

while (float(t) < 1.0):
    # no need to update BC since they are all zero for each stage for each time..
    if float(t) + float(dt) > 1.0:
        dt.assign(1.0 - float(t))
    solver.solve()

    for i in range(ns):
        u += float(dt) * BT.b[i] * ks[i]

    t.assign(float(t) + float(dt))  # takes a new value, not a Constant
    print(float(t))

print()
print(errornorm(uexact, u)/norm(uexact))
