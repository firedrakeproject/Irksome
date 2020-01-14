from firedrake import *
import numpy

N = 4
msh = UnitSquareMesh(N, N)

V = FunctionSpace(msh, "CG", 1)
x, y = SpatialCoordinate(msh)

# Stores initial condition!
u = interpolate(2.0 + sin(pi*x)*sin(pi*y), V)
unew = TrialFunction(V)
v = TestFunction(V)

dt = Constant(0.1)

a = dt * inner(grad(unew), grad(v))*dx + inner(unew, v)*dx
L = inner(u, v)*dx

uu = Function(V)

t = Constant(0.0)

#params = {"mat_type": "aij",
#          "ksp_type": "preonly",
#          "pc_type": "lu"}

solve(a==L, uu)

#    print(u.dat.data)
print(assemble(u**2*dx))
print(assemble(uu**2*dx))

    



