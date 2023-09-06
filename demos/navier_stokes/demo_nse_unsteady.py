from firedrake import *  # noqa: F403
from irksome import GaussLegendre, getForm, Dt, MeshConstant, TimeStepper
import numpy


ButcherTableau = GaussLegendre(1)

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

#Space
msh = mh[-1]


V = VectorFunctionSpace(msh, "CG", 4)
W = FunctionSpace(msh, "DG", 3)
Z = V * W

v, w = TestFunctions(Z)
x, y = SpatialCoordinate(msh)

up = Function(Z)
u, p = split(up)

n= FacetNormal(msh)

MC = MeshConstant(msh)
dt = MC.Constant(1.0/1600)
t = MC.Constant(0.0)
nu = Constant(0.001)
rho=1.0
r=0.05
Umean=1.0
L=0.1

# define the variational form once outside the loop
gamma = Constant(1.e2)
F = (inner(Dt(u), v) * dx
     +inner(dot(grad(u), u), v) * dx
     + gamma * inner(cell_avg(div(u)), cell_avg(div(v))) * dx
     + nu * inner(grad(u), grad(v)) * dx
     - inner(p, div(v)) * dx
     + inner(div(u), w) * dx)

# boundary conditions are specified for each subspace
UU = 1.5 * sin(pi * t / 8)
bcs = [DirichletBC(Z.sub(0),
                   as_vector([4 * UU * y * (y1 - y) / (y1**2), 0]), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 3, 5))]

#nullspace = MixedVectorSpaceBasis(
#    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

s_param={'ksp_type': 'preonly',
         'pc_type': 'lu',
         'pc_factor_shift_type': 'inblocks',
         'mat_type': 'aij',
         'snes_max_it': 100}
#'snes_monitor': None}
# the TimeStepper object builds the UFL for the multi-stage RK method
# and sets up a solver.  It also provides a method for updating the solution
# at each time step.
# It overwites the UFL constant t with the current time value at each
# step.  u is also updated in place.
stepper = TimeStepper(F, ButcherTableau, t, dt, up, bcs=bcs,
                      solver_parameters=s_param)

times = []
CD = []
CL = []

ut1 = dot(u, as_vector([n[1], -n[0]]))

while (float(t) < 1.0):
    # Update step
    stepper.advance()

    # Numerical quantities
    FD= assemble((nu*dot(grad(ut1), n)*n[1] - p*n[0])*ds(5) )
    FL= assemble(-(nu*inner(grad(ut1), n)*n[0]+p*n[1])*ds(5) )

    CD+= [-2 * FD / (Umean**2) / L] #Drag coefficient
    CL+= [-2 * FL / (Umean**2) / L] #Lift coefficient

    t.assign(float(t) + float(dt))

    print(float(t), CD[-1], CL[-1])

    times.append(float(t))


times = numpy.asarray(times)
CD = numpy.asarray(CD)
CL = numpy.asarray(CL)

iCD = numpy.argmax(CD)
tCDmax = times[iCD]
CDmax = CD[iCD]

iCL = numpy.argmax(CL)
tCLmax = times[iCL]
CLmax = CL[iCL]

#p_func=interpolate(p, W)
#p1=p_func( [0.15,0.2] )
#p2=p_func( [0.25,0.2] )
#pdiff= p1-p2

#Results
print("Level", lvl)
print("CD_max", CDmax)
print("t(CD_max)", tCDmax)
print("CL_max", CLmax)
print("t(CL_max)", tCLmax)

#u, p = up.split()
#print(p((0.15, .2)) - p((0.250, 0.2)))
