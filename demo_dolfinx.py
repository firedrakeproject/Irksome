from mpi4py import MPI
import dolfinx
import ufl
from irksome import GaussLegendre, Dt, MeshConstant
from irksome.stage_derivative import getForm
from irksome.tools import get_stage_space
from ufl import pi, atan, div, grad, inner, dx

butcher_tableau = GaussLegendre(2)
N = 64

x0 = 0.0
x1 = 10.0
y0 = 0.0
y1 = 10.0

msh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[x0, y0], [x1, y1]], [N, N])
V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
x, y = ufl.SpatialCoordinate(msh)

MC = MeshConstant(msh, backend="dolfinx")
dt = MC.Constant(10 / N)
t = MC.Constant(0.0)

Constant = lambda val: dolfinx.fem.Constant(msh, val)
S = Constant(2.0)
C = Constant(1000.0)


B = (
    (x - Constant(x0))
    * (x - Constant(x1))
    * (y - Constant(y0))
    * (y - Constant(y1))
    / C
)
R = (x * x + y * y) ** 0.5
uexact = B * atan(t) * (pi / 2.0 - atan(S * (R - t)))

rhs = Dt(uexact) - div(grad(uexact))

u = dolfinx.fem.Function(V)
u.interpolate(dolfinx.fem.Expression(uexact, V.element.interpolation_points))

v = ufl.TestFunction(V)
F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - inner(rhs, v) * dx

bc = []
# bc = DirichletBC(V, 0, "on_boundary")

# Get the function space for the stage-coupled problem and a function to hold the stages we're computing::

Vbig = get_stage_space(V, butcher_tableau.num_stages, backend="dolfinx")
k = dolfinx.fem.Function(Vbig)

# Get the variational form and bcs for the stage-coupled variational problem::

Fnew, bcnew = getForm(F, butcher_tableau, t, dt, u, k, bcs=bc, backend="dolfinx")
