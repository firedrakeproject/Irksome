# Wave equation in mixed form
# u_t + grad(p) = 0
# p_t + div(u) = 0
#
# We are using a symplectic DIRK (Qin-Zhang)

from firedrake import *
from irksome import QinZhang, Dt, TimeStepper
import numpy

N = 10

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
dt = Constant(1.0/N)

# Symplectic 2-stage DIRK method, should conserve energy up to roundoff on
# linear problems.
butcher_tableau = QinZhang()

# DIRK methods have a lower-triangular Butcher matrix,
# so we can use a block lower-triangular preconditioner to solve the problem
# exactly without needing a GMRES iteration.

# The issue is that with a mixed problem, we have multiple function spaces
# for each stage.  For a two stage method, we have
# V * W * V * W
# 
# and we will solve a variational problem for stages
# (ku0, kp0, ku1, kp1)
#
# The algebraic system for the RK method couples both bits of stage 0 together
# and then both bits of stage 1, so we need to tell PETSc
# to treat this as a 2x2 system rather than a 4x4.


params = {"mat_type": "aij",
          "ksp_type": "preonly",
          "snes_type": "ksponly",
          "pc_type": "fieldsplit",
          "pc_fieldsplit_type": "multiplicative"}

# this shows how to do the blocking logic for a general
# number of stages in the RK method and general number of
# spaces in the mixed space.  In our case (2 stages, 2 fields), this
# results in options:
#
# "pc_fieldsplit_0_fields": "0,1"
# "pc_fieldsplit_1_fields": "2,3"

for s in range(butcher_tableau.num_stages):
    stages_cur = ",".join([str(i) for i in range(s*len(Z), (s+1)*len(Z))])
    params["pc_fieldsplit_%d_fields" % s] = stages_cur

# and now we just set up a direct method on each stage.  If you have
# a good solver for a single-stage method, you can replace this direct
# method with that.  And you can do a different method for each stage if you like.

per_field = {"ksp_type": "preonly",
             "pc_type": "lu"}
for i in range(butcher_tableau.num_stages):
    params["fieldsplit_%d" % i] = per_field


stepper = TimeStepper(F, butcher_tableau, t, dt, up0,
                      solver_parameters=params)

# We just take one step to show that energy is conserved, so we're probably
# solving the system exactly.
print(float(t), assemble(E))
stepper.advance()
print(float(t), assemble(E))



