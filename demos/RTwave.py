# Wave equation in mixed form
# u_t + grad(p) = 0
# p_t + div(u) = 0
# with homogeneous Dirichlet BC p=0 (which are weakly enforced in mixed methods)

from firedrake import *
from IRKsome import GaussLegendreButcherTableau, LobattoIIIAButcherTableau, getForm, BackwardEulerButcherTableau

N = 10

msh = UnitSquareMesh(N, N)
V = FunctionSpace(msh, "RT", 1)
W = FunctionSpace(msh, "DG", 0)
Z = V*W

v, w = TestFunctions(Z)
x, y = SpatialCoordinate(msh)
up0 = project(as_vector([0, 0, sin(pi*x)*sin(pi*y)]), Z)

upnew = Function(Z)

u0, p0 = split(up0)

F = inner(div(u0), w) * dx - inner(p0, div(v)) * dx

E = 0.5 * (inner(u0, u0)*dx + inner(p0, p0)*dx)

tc = 0.0
t = Constant(tc)
dtc = 1.0 / N
dt = Constant(dtc)

#BT = LobattoIIIAButcherTableau(2)
#BT = GaussLegendreButcherTableau(1)
BT = BackwardEulerButcherTableau()

num_stages = len(BT.b)
num_fields = len(Z)

bigF, k = getForm(F, BT, t, dt, up0)

params = {"mat_type": "aij",
          "ksp_type": "preonly",
          "pc_type": "lu"}

prob = NonlinearVariationalProblem(bigF, k)
solver = NonlinearVariationalSolver(prob, solver_parameters=params)

while (tc < 1.0):
    solver.solve()
    print(tc, assemble(E))
        
    up0 += dt * k

    tc += dtc
    t.assign(tc)


