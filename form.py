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

def getForm(a, butcher, dt):
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
            anew += dt * map_integrands.map_integrand_dags(mapper, a)

    return Vfs, anew


AGaussLeg = np.array([[0.25, 0.25 - np.sqrt(3) / 6],
                      [0.25+np.sqrt(3)/6, 0.25]])



if __name__ == "__main__":
    N = 8
    msh = UnitSquareMesh(N, N)
    hierarchy = MeshHierarchy(msh, 4)
    msh = hierarchy[-1]
    V = FunctionSpace(msh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    # good Helmholtz so we can use Neumann BC for now
    a = inner(grad(u), grad(v))*dx + inner(u, v)*dx

    dt = Constant(0.1)

    Vfs, anew = getForm(a, AGaussLeg, dt)

    F = Function(Vfs)
    L = inner(F, anew.arguments()[0])*dx

    uu = Function(Vfs)
    with uu.dat.vec_wo as x:
        x.setRandom()

    params = {"mat_type": "aij",
              "ksp_monitor": None,
              "ksp_type": "gmres",
              "pc_type": "mg",
              "mg_levels": {
                  "ksp_type": "richardson",
                  "ksp_richardson_scale": 3/4,
                  "ksp_max_it": 2,
                  "ksp_monitor_true_residual": None,
                  "ksp_norm_type": "unpreconditioned",
                  "pc_type": "pbjacobi"}
    }

    solve(anew==L, uu, solver_parameters=params)








