# Wave equation in mixed form
# u_t + grad(p) = 0
# p_t + div(u) = 0
# with homogeneous Dirichlet BC p=0 (which are weakly enforced in mixed methods)

from firedrake import *
from IRKsome import GaussLegendre, Dt, AdaptiveTimeStepper

N = 64

msh = UnitSquareMesh(N, N)
V = FunctionSpace(msh, "RT", 1)
W = FunctionSpace(msh, "DG", 0)
Z = V*W

v, w = TestFunctions(Z)
x, y = SpatialCoordinate(msh)
up0 = project(as_vector([0, 0, sin(pi*x)*sin(pi*y)]), Z)

u0, p0 = split(up0)

F = inner(Dt(u0), v)*dx + inner(div(u0), w) * dx + inner(Dt(p0), w)*dx - inner(p0, div(v)) * dx

E = 0.5 * (inner(u0, u0)*dx + inner(p0, p0)*dx)

t = Constant(0.0)
dt = Constant(1.0 / N)

butcher_tableau = GaussLegendre(2)

# the TimeStepper object builds the UFL for the multi-stage RK method
# and sets up a solver.  It also provides a method for updating the solution
# at each time step.
# It overwites the UFL constant t with the current time value at each
# step.  u is also updated in place.

params = {"mat_type": "aij",
          "snes_type": "ksponly",
          "ksp_type": "gmres",
          "pc_type": "lu"}

stepper = AdaptiveTimeStepper(F, butcher_tableau, t, dt, up0,
                              tol=1.e-3, dtmin=1.e-5,
                              solver_parameters=params)

initial_energy = assemble(E)
while (float(t) < 1.0):
    err = stepper.advance()
    print(float(t), float(dt), assemble(E))

    t.assign(float(t) + float(dt))
final_energy = assemble(E)

print("Energy difference: ", abs(final_energy - initial_energy))
