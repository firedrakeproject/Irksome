from firedrake import *  # noqa: F403

from ufl.algorithms.ad import expand_derivatives

from IRKsome import LobattoIIIC, getForm, Dt, TimeStepper

BT = LobattoIIIC(3)
ns = len(BT.b)

# Single point of entry in case you want to change the size of the box.
x0 = 0.0
x1 = 2.2
y0 = 0.0
y1 = 0.41


#Mesh
stepfile = "circle.step"
h = 0.05
order = 2
lvl=2
mh = OpenCascadeMeshHierarchy(stepfile, element_size=h, levels=lvl, order=order, cache=False, verbose=True)



#Space
msh = mh[-1]
V = VectorFunctionSpace(msh, "CG", 2)
W = FunctionSpace(msh, "CG", 1)
Z = V * W
num_fields=len(Z)
v, w = TestFunctions(Z)
x, y = SpatialCoordinate(msh)
up = Function(Z)
u, p = split(up)
n= FacetNormal(msh)
E = 0.5 * (inner(u, u)*dx + inner(p, p)*dx)

# Stores initial condition!
tc = 0
dtc = 1. / 1600
dt = Constant(dtc)
t = Constant(0.0)
nu=0.001
rho=1.0
r=0.05
Umean=1.0
L=0.1

# define the variational form once outside the loop
F = inner(Dt(u), v) * dx + inner(dot(u, grad(u)), v) * dx + nu*inner(grad(u), grad(v)) * dx - inner(p, div(v)) * dx \
    -inner( p*n,v )*ds(2) #do-nothing boundary condition 
    #+inner(div(u), w) * dx \

# boundary conditions are specified for each subspace
bcs_V = [ DirichletBC(Z.sub(0), as_vector( (4*1.5*sin(pi*t/8)*(y1-y)/(y1**2),0)) ,(4,)), DirichletBC(Z.sub(0), Constant((0,0)), (1,3,5)) ] 
bcs_W = [ ] 
bcs=[bcs_V,bcs_W]

#nullspace = MixedVectorSpaceBasis(
#    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# We only need to set up the solver one time!
s_param={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_shift_type': 'inblocks'}


# the TimeStepper object builds the UFL for the multi-stage RK method
# and sets up a solver.  It also provides a method for updating the solution
# at each time step.
# It overwites the UFL constant t with the current time value at each
# step.  u is also updated in place.
stepper = TimeStepper(F, BT, t, dt, up, bcs=bcs,
                      solver_parameters=s_param)

b = BT.b

CD=[]
CL=[]
while (float(t) < 0.01):
    #Update step
    print(tc, assemble(up[0]*dx))
    stepper.advance()
    print(float(t))
    t.assign(float(t) + float(dt))

    #Numerical quantities
    ut1=dot(u, as_vector( (n[1],-n[0]) ))
    FD= assemble(  (nu*inner(grad(ut1),n)*n[1]-p*n[0])*ds )
    FL= assemble(  -(nu*inner(grad(ut1),n)*n[0]-p*n[1])*ds )
    
    CD+= [(2*FD/(Umean**2*L),float(t)) ] #Drag coefficient
    CL+= [(2*FL/(Umean**2*L),float(t)) ] #Lift coefficient
    
    tc += dtc
    t.assign(tc)


p_func=interpolate(p, W)
#p1=p_func( [0.15,0.2] )
#p2=p_func( [0.25,0.2] )
#pdiff= p1-p2

#Results
CDmax= max(CD, key=lambda item:item[0])[0]
tCDmax= max(CD, key=lambda item:item[0])[1]

CLmax= max(CL, key=lambda item:item[0])[0]
tCLmax= max(CL, key=lambda item:item[0])[1]

print("Level", lvl)
print("CD_max", CDmax)
print("t(CD_max)", tCDmax)
print("CL_max", CLmax)
print("t(CL_max)", tCLmax)

print(CD)
print(CL)
#print("pdiff", pdiff)