from firedrake import *
from irksome import Dt, TimeStepper, GaussLegendre

nx = 32
re = 100
dt = 0.05
stages = 3

mesh = UnitIntervalMesh(nx)
V = VectorFunctionSpace(mesh, "CG", 2)
R = FunctionSpace(mesh, "R", 0)
t = Function(R).assign(0)
dt = Function(R).assign(dt)

x, = SpatialCoordinate(mesh)
ic = 1 + sin(2*pi*x)

u = Function(V, name="u").project(as_vector([ic]))
v = TestFunction(V)

nu = Constant(1/re)

F = (
    inner(Dt(u), v)*dx
    + inner(dot(u, nabla_grad(u)), v)*dx
    + inner(nu*grad(u), grad(v))*dx
)

zero = Constant(0)
bcs = [DirichletBC(V, as_vector([zero]), "on_boundary")]
newton_params = {
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
}

params = {
    "snes": {
        "converged_reason": None,
        "monitor": None,
        "rtol": 1e-5,
    },
    "snes_type": "python",
    "snes_python_type": "irksome.SDCLDSNES",
    "sdc": newton_params,
    "sdc": {
        "snes_type": "python",
        "snes_python_type": "firedrake.FieldsplitSNES",
        "snes_fieldsplit_type": "multiplicative",
        "fieldsplit_0": newton_params,
        "fieldsplit_1": newton_params,
        "fieldsplit_2": newton_params,
    }
}

stepper = TimeStepper(
    F, GaussLegendre(3), t, dt, u,
    bcs=bcs, solver_parameters=params)

stepper.advance()
