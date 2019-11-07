from firedrake import *

import numpy as np

from ufl.algorithms import MultiFunction, map_integrands

class ArgumentReplacer(MultiFunction):
    def __init__(self, arg_map):
        self.arg_map = arg_map
        super(ArgumentReplacer, self).__init__()

    expr = MultiFunction.reuse_if_untouched

    def argument(self, o):
        return self.arg_map[o]

def getForm(a, butcher, dt, un):
    # a is the bilinear form for the elliptic part of the operator
    # (i.e. no time derivative)
    # butcher is the A matrix from the Butcher tableau
    # this creates the big bilinear form for computing all of the
    # RK stages
    #
    # dt is a Constant holding the time step
    #
    # To really do time-stepping, we would need the rest of the tableu
    # to set up the right-hand side and take the weighted combination
    # of the stages.  We can do that later.

    # un is a Function holding data for the current time step value
    
    test, trial = a.arguments()
    assert test.function_space() == trial.function_space()
    fs = test.function_space()
    fel = fs.ufl_element()
    msh = fs.mesh()

    num_stages = butcher.shape[0]

    Vfs = VectorFunctionSpace(msh, fel, dim=num_stages)

    unew = TrialFunction(Vfs)
    vnew = TestFunction(Vfs)

    anew = inner(unew, vnew)*dx

    for i in range(num_stages):
        for j in range(num_stages):
            mapper = ArgumentReplacer({test: vnew[i],
                                       trial: unew[j]})
            anew += dt * Constant(butcher[i, j]) * map_integrands.map_integrand_dags(mapper, a)

    aaction = action(a, un)
    L = 0
    for i in range(num_stages):
        mapper = ArgumentReplacer({test: vnew[i]})
        L -= map_integrands.map_integrand_dags(mapper, aaction)
    
            
    return Vfs, anew, L


AGaussLeg4 = np.array([[0.25, 0.25 - np.sqrt(3) / 6],
                       [0.25+np.sqrt(3)/6, 0.25]])

Radau35 = np.array([[11/45 - 7*sqrt(6)/360, 37/225 - 169*sqrt(6)/1800, -2/225 + sqrt(6)/75],
                    [37/225 - 169*sqrt(6)/1800, 11/45 - 7*sqrt(6)/360, -2/225 - sqrt(6)/75],
                    [4/9 - sqrt(6)/36, 4/9 + sqrt(6)/36, 1/9]])

AGaussLeg6 = np.array([[5./36, 2/9 - 1./np.sqrt(15), 5./36 - np.sqrt(15)/30],
                       [5./36+np.sqrt(15)/24, 2./9, 5./36 - np.sqrt(15)/24],
                       [5./36 + np.sqrt(15)/30, 2./9 + np.sqrt(15)/15, 5./36]])

if __name__ == "__main__":
    N = 16
    msh = UnitSquareMesh(N, N)
    hierarchy = MeshHierarchy(msh, 1)
    msh = hierarchy[-1]
    V = FunctionSpace(msh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    # good Helmholtz so we can use Neumann BC for now
    a = inner(grad(u), grad(v))*dx + inner(u, v)*dx

    x, y = SpatialCoordinate(msh)
    u0 = project(sin(pi*x)*sin(pi*y), V)
    u1 = Function(V)
    
    dt = Constant(0.1)

    Vfs, anew, Lnew = getForm(a, Radau35, dt, u0)

    params = {"mat_type": "aij",
              "ksp_type": "preonly",
              "pc_type": "lu"}
    # params = {"mat_type": "aij",
    #           "ksp_monitor": None,
    #           "ksp_type": "gmres",
    #           "pc_type": "mg",
    #           "mg_levels": {
    #               "ksp_type": "richardson",
    #               "ksp_richardson_scale": 3/4,
    #               "ksp_max_it": 2,
    #               "ksp_monitor_true_residual": None,
    #               "ksp_norm_type": "unpreconditioned",
    #               "pc_type": "pbjacobi"}
    # }

    k = Function(Vfs)
    
    solve(anew==Lnew, k, solver_parameters=params)

    u1.dat.data[:] = u0.dat.data[:]
    print(u1.dat.data.shape)
    print(k.dat.data.shape)
    # hack: u1.dat.data[:] = u0.dat.data[:]
    # u1.dat.data[:] += dt * k.dat.data @ b from Butcher table
    # loop over stages, add dt times k[i] into u1







