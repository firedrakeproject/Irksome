from ufl import replace

from firedrake import *  # noqa: F403

from ufl.algorithms.ad import expand_derivatives

from IRKsome import LobattoIIIA, getForm



BT = LobattoIIIA(2)

ns = len(BT.b)
N = 32
msh = UnitSquareMesh(N, N)

V = FunctionSpace(msh, "CG", 1)
x, y = SpatialCoordinate(msh)
tc=0
dtc=1.0/10
dt = Constant(dtc)
t = Constant(0.0)

# Stores initial condition
uexact = exp(-t)*cos(pi*x)*sin(pi*y)

# MMS works on symbolic differentiation of true solution, not weak form
# Except we might need to futz with this since replacement is breaking on this!
rhs = expand_derivatives(diff(uexact,t) - div(grad(uexact)))




u = interpolate(uexact, V)

# Build it once and update it rather than deepcopying in the
# time stepping loop.
unew = Function(V)
unew = interpolate(exp(0)*cos(pi*x)*sin(pi*y), V)

# define the variational form once outside the loop
# notice that there is no time derivative term.  Our function
# supplies that.
v = TestFunction(V)

F = inner(grad(u), grad(v))*dx - inner(rhs, v)*dx
bc= DirichletBC(V, uexact, "on_boundary" ) #Taking the exact solution for boundary conditions

Fnew, k, bcnew = getForm(F, BT, t, dt, u, bc) 
bcnew=None

fs = Fnew.arguments()[0].function_space()

params = {"mat_type": "aij",
          "ksp_type": "preonly",
          "pc_type": "lu"}

#print(assemble(u**2*dx))
prob = NonlinearVariationalProblem(Fnew, k, bcs=bcnew)
solver = NonlinearVariationalSolver(prob, solver_parameters=params)

# get a tuple of the stages, each as a Coefficient
ks = k.split()

v = F.arguments()[0]
VV = v.function_space()
Vbig = numpy.prod([VV for i in range(ns)])    

while (tc<2):
    g1 = interpolate(-exp(-(t + 0 * dt))*cos(pi*x)*sin(pi*y), V)
    g2 = interpolate(-exp(-(t + 1 * dt))*cos(pi*x)*sin(pi*y), V)
    
    bcnew=[DirichletBC(Vbig[0], g1 , "on_boundary" ),DirichletBC(Vbig[1], g2 , "on_boundary" )]
    prob = NonlinearVariationalProblem(Fnew, k, bcs=bcnew)
    solver = NonlinearVariationalSolver(prob, solver_parameters=params)
    solver.solve()

    # update unew
    for i in range(ns):
        unew += dtc * BT.b[i] * ks[i]

    u.assign(unew)

    tc+=dtc
    t.assign(tc)  # takes a new value, not a Constant
    #print(tc)

     #real solution evaluated at t=2

ueval = exp(-t)*cos(pi*x)*sin(pi*y)
print(assemble(ueval**2*dx))
print(assemble(unew**2*dx))
print(errornorm(ueval, unew))
plot(interpolate(ueval,V) )
plot(unew)
