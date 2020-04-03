from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import os
from IRKsome import Dt, getForm, BackwardEuler, LobattoIIIA

if not os.path.exists("pictures/cahnhilliard"):
    os.makedirs("pictures/cahnhilliard")
elif not os.path.isdir("pictures/cahnhilliard"):
    raise RuntimeError("Cannot create output directory, file of given name exists")

N = 16

msh = UnitSquareMesh(N, N)

# Some refined meshes for P1 visualisation
vizmesh = MeshHierarchy(msh, 2)[-1]

V = FunctionSpace(msh, "Bell", 5)

lmbda = Constant(1.e-2)
dt = Constant(5.0e-6)
T = Constant(0.0025)
t = Constant(0.0)
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

eFF = (inner(Dt(c), v) * dx +
       inner(M*grad(dfdc(c)), grad(v))*dx +
       inner(M*lmbda*lap(c), lap(v))*dx -
       inner(M*lmbda*lap(c), dot(grad(v), n))*ds -
       inner(M*lmbda*dot(grad(c), n), lap(v))*ds +
       inner(beta/h*M*lmbda*dot(grad(c), n), dot(grad(v), n))*ds)

# Crank-Nicolson, like in Kirby/Mitchell
# But takes some work since it's coded as a two-stage method
ButcherTableau = LobattoIIIA(2)  
Fnew, k, _, _ = getForm(eFF, ButcherTableau, t, dt, c)

prob = NonlinearVariationalProblem(Fnew, k)

params = {'snes_monitor': None, 'snes_max_it': 100,
          'snes_linesearch_type': 'l2',
          'ksp_type': 'preonly',
          'pc_type': 'lu', 'mat_type': 'aij',
          'pc_factor_mat_solver_type': 'mumps'}

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


import matplotlib.pyplot as plt
get_output()
cs = tripcolor(output, vmin=0, vmax=1)
plt.colorbar(cs)
plt.savefig('pictures/cahnhilliard/init.pdf', format='pdf', bbox_inches='tight', pad_inches=0)


ks = k.split()
while float(t) < float(T):
    PETSc.Sys.Print("Time: %s" % float(t))
    t.assign(float(t) + float(dt))
    solver.solve()
    for s in range(ButcherTableau.num_stages):
        c += float(dt) * ButcherTableau.b[s] * ks[s]
    
    #fl.write(get_output())

get_output()
cs = tripcolor(output, vmin=0, vmax=1)
plt.colorbar(cs)
plt.savefig('pictures/cahnhilliard/final.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

print(np.max(c.dat.data[::6]))
