from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import os
import IRKsome

if not os.path.exists("pictures/cahnhilliard"):
    os.makedirs("pictures/cahnhilliard")
elif not os.path.isdir("pictures/cahnhilliard"):
    raise RuntimeError("Cannot create output directory, file of given name exists")

N = 16

msh = UnitSquareMesh(N, N)

# Some refined meshes for P1 visualisation
vizmesh = MeshHierarchy(msh, 2)[-1]

V = FunctionSpace(msh, "Bell", 5)

lmbda = 1.e-2
delta_t = 5.0e-6
dt = Constant(delta_t)
t = 0.0
T = 0.0025
t_ufl = Constant(t)
M = Constant(1)

beta = Constant(250.0)

# set up initial condition
np.random.seed(42)
c = Function(V)
c.dat.data[::6] = 0.63 + 0.2*(0.5 - np.random.random(c.dat.data[::6].shape))

v = TestFunction(V)

def dfdc(cc):
    return 200*(cc*(1-cc)**2-cc**2*(1-cc))


def lap(u):
    return div(grad(u))


n = FacetNormal(msh)
h = CellSize(msh)

eFF = (inner(M*grad(dfdc(c)), grad(v))*dx +
       inner(M*lmbda*lap(c), lap(v))*dx -
       inner(M*lmbda*lap(c), dot(grad(v), n))*ds -
       inner(M*lmbda*dot(grad(c), n), lap(v))*ds +
       inner(beta/h*M*lmbda*dot(grad(c), n), dot(grad(v), n))*ds)

#BT = IRKsome.BackwardEuler()
BT = IRKsome.GaussLegendre(2)
Fnew, k = IRKsome.getForm(eFF, BT, t_ufl, dt, c)

prob = NonlinearVariationalProblem(Fnew, k)

params = {'snes_monitor': None, 'snes_max_it': 100,
          'snes_linesearch_type': 'basic',
          'ksp_type': 'preonly',
          'pc_type': 'lu', 'mat_type': 'aij'}

solver = NonlinearVariationalSolver(prob, solver_parameters=params)

output = Function(FunctionSpace(vizmesh, "P", 1),
                  name="concentration")

P5 = Function(FunctionSpace(msh, "P", 5))
proj = Projector(c, P5)
intp = Interpolator(c, P5)


def project_output():
    proj.project()
    return prolong(P5, output)


def interpolate_output():
    intp.interpolate()
    return prolong(P5, output)


use_interpolation = True
if use_interpolation:
    get_output = interpolate_output
else:
    get_output = project_output

# fl = File("pictures/cahnhilliard/ch.pvd")
# fl.write(get_output())


def surfplot(name):
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    output = get_output()
    mesh = output.ufl_domain()
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.set_aspect("equal")
    axes.axis("off")
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    x = mesh.coordinates.dat.data_ro[:, 0]
    y = mesh.coordinates.dat.data_ro[:, 1]
    cells = mesh.coordinates.cell_node_map().values
    triangulation = Triangulation(x, y, cells)
    cs = axes.tripcolor(triangulation,
                        np.clip(output.dat.data_ro, 0, 1),
                        cmap=plt.cm.gray,
                        edgecolors='none', vmin=0, vmax=1,
                        shading="gouraud")
    plt.colorbar(cs)
    axes = plot(mesh, surface=True, axes=axes)
    axes.collections[-1].set_color("black")
    axes.collections[-1].set_linewidth(1)

    plt.savefig('pictures/cahnhilliard/{name}.pdf'.format(name=name),
                format='pdf', bbox_inches='tight', pad_inches=0)


surfplot("initial")
while t < T:
    PETSc.Sys.Print("Time: %s" % t)
    t += delta_t
    solver.solve()
    if BT.num_stages == 1:
        c += delta_t * k
    else:
        for s in range(BT.num_stages):
            c.dat.data[:] += delta_t * BT.b[s] * k.dat.data[s][:]
    
    # fl.write(get_output())
surfplot("final")
print(np.max(c.dat.data[::6]))
