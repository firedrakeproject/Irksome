# suggested setting OMP_NUM_THREADS=1 to improve performance
import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from firedrake.__future__ import interpolate
from irksome import MeshConstant, Dt, RadauIIA, TimeStepper
from ufl import sin, cos, pi, as_vector, exp, div, dot
import numpy as np

from KronPC import KronPC  # import the preconditioner


mesh_size = 4
nu = 1.0 / 50.0
T = 2.0
s = 2
dt_value = T / 10.0
gamma = Constant(50.0)

mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size, quadrilateral=True)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

t = Constant(0.0)
MC = MeshConstant(mesh)
dt = MC.Constant(dt_value)

x = SpatialCoordinate(mesh)
u_exact = 0.5 * exp(T - t) * as_vector([
    sin(pi * x[0]) * cos(pi * x[1]),
    -cos(pi * x[0]) * sin(pi * x[1])
])
p_exact = Constant(0.0)

u_t = Dt(u_exact)
Delta_u = div(grad(u_exact))
conv_u = dot(grad(u_exact), u_exact)
f_expr = u_t - nu * Delta_u - conv_u

w = Function(W)
(u, p) = split(w)
(u0, p0) = w.subfunctions
u0.interpolate(u_exact)
p0.assign(0.0)

bc = DirichletBC(W.sub(0), u_exact, "on_boundary")

v, q = TestFunctions(W)
F = (inner(Dt(u), v) * dx
        + nu * inner(grad(u), grad(v)) * dx
        + inner(dot(grad(u), u), v) * dx
        - inner(p, div(v)) * dx
        + inner(div(u), q) * dx
        - inner(f_expr, v) * dx
        + gamma * inner(div(u), div(v)) * dx)

nsp = [(1, VectorSpaceBasis(constant=True))]

velocity_fields = ",".join(str(2*i) for i in range(s))
pressure_fields = ",".join(str(2*i+1) for i in range(s))

parameters = {
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-8,
    "ksp_converged_reason": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "upper",
    "pc_fieldsplit_0_fields": velocity_fields,
    "pc_fieldsplit_1_fields": pressure_fields,
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "ksp_rtol": 1e-10,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_ksp_type": "preonly",
        "assembled_pc_type": "lu",
    },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        "ksp_rtol": 1e-10,
        "pc_type": "python",
        "pc_python_type": "KronPC.KronPC",
        "kron": {
            "operator": "mass",
            "pow": 1,
            "pc_type": "lu"
        }
    },
}

stepper = TimeStepper(
    F,
    RadauIIA(s),
    t,
    dt,
    w,
    bcs=bc,
    stage_type="deriv",
    solver_parameters=parameters,
    nullspace=nsp
)

while float(t) < T - 1e-10:
    stepper.advance()
    t.assign(float(t) + float(dt))

steps, nonlinear_its, linear_its = stepper.solver_stats()
print("Total number of timesteps was %d" % (steps))
print("Average number of nonlinear iterations per timestep was %.2f" % (nonlinear_its/steps))
print("Average number of linear iterations per timestep was %.2f" % (linear_its/steps))
