from firedrake import *  # noqa: F403
import numpy


class DGMassInv(PCBase):

    def initialize(self, pc):
        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        V = dmhooks.get_function_space(pc.getDM())
        # get function spaces
        u = TrialFunction(V)
        v = TestFunction(V)
        massinv = assemble(Tensor(inner(u, v)*dx).inv)
        self.massinv = massinv.petscmat
        self.nu = appctx["nu"]
        self.gamma = appctx["gamma"]

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        self.massinv.mult(x, y)
        scaling = float(self.nu) + float(self.gamma)
        y.scale(-scaling)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("Sorry!")


# Corners of the box.
x0 = 0.0
x1 = 2.2
y0 = 0.0
y1 = 0.41

#Mesh
stepfile = "circle.step"
# Close to what Turek's time-dependent benchmark has on coarsest mesh
h = 0.1 
order = 2
lvl = 3
mh = OpenCascadeMeshHierarchy(stepfile, element_size=h,
                              levels=lvl, order=order, cache=False,
                              verbose=True)

msh = mh[-1]

V = VectorFunctionSpace(msh, "CG", 2)
W = FunctionSpace(msh, "DG", 0)
Z = V * W

v, w = TestFunctions(Z)
x, y = SpatialCoordinate(msh)

up = Function(Z)
u, p = split(up)

n= FacetNormal(msh)

nu = Constant(0.001)
rho=1.0
r=0.05
Umean=0.2
L=0.1

# define the variational form once outside the loop
gamma = Constant(1.e2)
F = (inner(dot(grad(u), u), v) * dx
     + gamma * inner(cell_avg(div(u)), cell_avg(div(v))) * dx
     + nu * inner(grad(u), grad(v)) * dx
     - inner(p, div(v)) * dx
     + inner(div(u), w) * dx)

# boundary conditions are specified for each subspace
UU = Constant(0.3)
bcs = [DirichletBC(Z.sub(0),
                   as_vector([4 * UU * y * (y1 - y) / (y1**2), 0]), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 3, 5))]


s_param = {'snes_monitor': None,
           'ksp_type': 'fgmres',
           #'ksp_view': None,
           'ksp_monitor': None,
           'pc_type': 'fieldsplit',
           'pc_fieldsplit_type': 'schur',
           'pc_fieldsplit_schur_fact_type': 'full',
           'fieldsplit_0': {
               'ksp_type': 'preonly',
               'pc_type': 'lu'},
           'fieldsplit_1': {
               'ksp_type': 'gmres',
               'ksp_max_it': 1,
               'pc_type': 'python',
               'pc_python_type': '__main__.DGMassInv'
               }}


appctx = {'nu': nu, 'gamma': float(gamma), 'velocity_space': 0}
ut1 = dot(u, as_vector([n[1], -n[0]]))

solve(F==0, up, bcs=bcs, solver_parameters=s_param, appctx=appctx)

FD= assemble((nu*dot(grad(ut1), n)*n[1] - p*n[0])*ds(5) )
FL= assemble(-(nu*inner(grad(ut1), n)*n[0]+p*n[1])*ds(5) )

CD+= [-2 * FD / (Umean**2) / L] #Drag coefficient
CL+= [-2 * FL / (Umean**2) / L] #Lift coefficient

print("Level", lvl)
print("CD", CD)
print("CL", CL)

#u, p = up.split()
#print(p((0.15, .2)) - p((0.250, 0.2)))

# u, p = up.split()
# File("nse_steady.pvd").write(u, p)
