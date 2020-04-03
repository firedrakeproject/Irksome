# Wave equation in mixed form
# u_t + grad(p) = 0
# p_t + div(u) = 0
# with homogeneous Dirichlet BC p=0 (which are weakly enforced in mixed methods)

from firedrake import *
from IRKsome import GaussLegendre, getForm, Dt

N = 64

msh = UnitSquareMesh(N, N)
V = FunctionSpace(msh, "RT", 2)
W = FunctionSpace(msh, "DG", 1)
Z = V*W

v, w = TestFunctions(Z)
x, y = SpatialCoordinate(msh)
up0 = project(as_vector([0, 0, sin(pi*x)*sin(pi*y)]), Z)

u0, p0 = split(up0)

F = inner(Dt(u0), v)*dx + inner(div(u0), w) * dx + inner(Dt(p0), w)*dx - inner(p0, div(v)) * dx

E = 0.5 * (inner(u0, u0)*dx + inner(p0, p0)*dx)

t = Constant(0.0)
dt = Constant(10 / N)

BT = GaussLegendre(2)

b = BT.b
num_fields = len(Z)

Fnew, k, bcnew, bcdata = getForm(F, BT, t, dt, up0)

params = {"mat_type": "aij",
          "ksp_type": "preonly",
          "pc_type": "lu"}

prob = NonlinearVariationalProblem(Fnew, k)
solver = NonlinearVariationalSolver(prob, solver_parameters=params)

while (float(t) < 1.0):
    if (float(t) + float(dt) > 1.0):
        dt.assign(1.0 - float(t))

    solver.solve()
    print(float(t), assemble(E))

    for s in range(BT.num_stages):
        for i in range(num_fields):
            up0.dat.data[i][:] += float(dt) * b[s] * k.dat.data[num_fields*s+i][:]

    t.assign(float(t) + float(dt))

print(float(t), assemble(E))


