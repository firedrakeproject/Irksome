# Wave equation in mixed form
# u_t + grad(p) = 0
# p_t + div(u) = 0
# with homogeneous Dirichlet BC p=0 (which are weakly enforced in mixed methods)

from firedrake import *
from IRKsome import GaussLegendreButcherTableau

N = 10

msh = UnitSquareMesh(N, N)
V = FunctionSpace(msh, "RT", 1)
W = FunctionSpace(msh, "DG", 0)
Z = V*W

v, w = TestFunctions(Z)
x, y = SpatialCoordinate(msh)
up0 = project(as_vector([0, 0, sin(pi*x)*sin(pi*y)]), Z)

u0, p0 = split(up0)

kup = Function(Z)

tc = 0.0
t = Constant(tc)
dtc = 1.0 / N
dt = Constant(dtc)

E = 0.5 * (inner(u0, u0)*dx + inner(p0, p0)*dx)


params = {"mat_type": "aij",
          "ksp_type": "preonly",
          "pc_type": "lu"}

bt = GaussLegendreButcherTableau(1)
A = bt.A

ku, kp = split(kup)

F = (inner(ku, v) * dx + inner(kp, w) * dx
     + inner(div(u0 + dt * A[0, 0] * ku), w) * dx
     - inner(p0 + dt * A[0, 0] * kp, div(v)) * dx)

prob = NonlinearVariationalProblem(F, kup)
solver = NonlinearVariationalSolver(prob, solver_parameters=params)

while (tc < 1.0):
    solver.solve()
    print(tc, assemble(E))
        
    up0 += dt * kup

    tc += dtc
    t.assign(tc)

    
