# Wave equation in mixed form
# u_t + grad(p) = 0
# p_t + div(u) = 0
# with homogeneous Dirichlet BC p=0 (which are weakly enforced in mixed methods)

from firedrake import *

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
unew, pnew = split(upnew)

tc = 0.0
t = Constant(tc)
dtc = 1.0 / N
dt = Constant(dtc)


F = (inner((unew-u0)/dt, v)*dx + inner((pnew-p0)/dt, w)*dx
     + inner(div(0.5*(u0+unew)), w) * dx
     - inner(0.5*(p0+pnew), div(v)) * dx)

E = 0.5 * (inner(u0, u0)*dx + inner(p0, p0)*dx)


params = {"mat_type": "aij",
          "ksp_type": "preonly",
          "pc_type": "lu"}

prob = NonlinearVariationalProblem(F, upnew)
solver = NonlinearVariationalSolver(prob, solver_parameters=params)


while (tc < 1.0):
    solver.solve()
    print(tc, assemble(E))
        
    up0.assign(upnew)

    tc += dtc
    t.assign(tc)
    print(tc, assemble(E))

    
