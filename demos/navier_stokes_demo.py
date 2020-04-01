from firedrake import *  # noqa: F403

from ufl.algorithms.ad import expand_derivatives

from IRKsome import GaussLegendre, getForm, Dt

BT = LobattoIIIC(3)
ns = len(BT.b)
N = 128
coarseN = 8 # size of coarse grid

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
V = VectorFunctionSpace(M, "CG", 4)
W = FunctionSpace(M, "CG", 3)
Z = V * W
v, w = TestFunctions(Z)
x, y = SpatialCoordinate(msh)
up = Function(Z)
u, p = split(up)
n= FacetNormal(mesh)

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
F = inner(Dt(u), v) * dx + inner(dot(u, grad(u)), v) * dx + nu*inner(grad(u), grad(v)) * dx - inner(p, div(v)) * dx 
    #+inner(div(u), w) * dx
    - inner( dot(p,n),v )*ds #do-nothing boundary condition 

U = 1.5*sin(pi*t/8)
Ut = interpolate(U, V)
bcs = [ DirichletBC(Z.sub(0), (4*Ut*(y1-y)/(y1^2),0) , (4,)) #parabolic inflow PENDING
        DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))] #no-slip boundary conditions PENDING

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# hand off the nonlinear function F to get weak form for RK method
Fnew, k, bcnew, bcdata = getForm(F, BT, t, dt, u, bcs=bcs)

# We only need to set up the solver one time!
s_param={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_shift_type': 'inblocks'}



prob = NonlinearVariationalProblem(Fnew, k, bcs=bcnew)
solver = NonlinearVariationalSolver(prob, solver_parameters=s_param)

# get a tuple of the stages, each as a Coefficient
ks = k.split()

CDmax=-1
tCDmax=0
CLmax=-1
tCLmax=0
while (tc < 8.0):
    solver.solve()

    for s in range(BT.num_stages):
        for i in range(num_fields):
            up.dat.data[i][:] += dtc * b[s] * k.dat.data[num_fields*s+i][:]

    #sigma= nu*grad(u)-p

    #Numerical quantities
    ut1=dot(u), (n[1],-n[0])
    FD= assemble(  (nu*inner(grad(u),n)*n[1]-p*n[0])*ds )
    FL= assemble(  -(nu*inner(grad(u),n)*n[0]-p*n[1])*ds )
    
    CD= 2*FD/(Umean**2*L) #Drag coefficient
    CL= 2*FD/(Umean**2*L) #Lift coefficient
    if( CD>CDmax): 
        CDmax=CD
        tCDmax=t

    if( CL>CLmax): 
        CLmax=CL
        tCLmax=t

    #Update t for next step
    tc += dtc
    t.assign(tc)

p_func=interpolate(p, W)
p1=p(0.15,0.2)
p2=p(0.25,0.2)
pdiff= p1-p2

#Results
print("Level", lvl)
print("CD_max", CDmax)
print("t(CD_max)", tCDmax)
print("CL_max", CLmax)
print("t(CL_max)", tCLmax)
print("pdiff", pdiff)