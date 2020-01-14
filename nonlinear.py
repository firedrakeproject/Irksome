from firedrake import *   # noqa F403

import numpy
from ufl import replace
import butcher


def getForm(F, butcher, t, dt, u0):
    # F: UFL
    # butcher: object with properties .A, .b, and .c
    # t and dt: Constant
    # u0: Function in the function space, current state.
    """Given a variational form F(u; v) describing a nonlinear
    time-dependent problem (u_t, v, t) + F(u, t; v) and a
    Butcher tableau, produce UFL for the s-stage RK method."""

    test, = F.arguments()

    fs = test.function_space()
    assert fs == u0.function_space()

    A = butcher.A
    c = butcher.c

    num_stages = A.shape[0]

    # TODO: think about VectorFunctionSpace, which is more efficient
    # but breaks if V is itself a VectorFunctionSpace, I think.
    Vbig = numpy.prod([fs for i in range(num_stages)])


    vnew = TestFunction(Vbig)
    k = Function(Vbig)

    Fnew = inner(k, vnew)*dx

    if num_stages > 1:
        for i in range(num_stages):
            unew = u0 + dt * numpy.sum([Constant(A[i, j]) * k[j] for j in range(num_stages)])
            tnew = t + Constant(c[i]) * dt
            Fnew += replace(F, {t: tnew,
                                u: unew,
                                v: vnew[i]})
    else:
        tnew = t + Constant(c[0])*dt
        unew = u0 + dt * Constant(A[0, 0]) * k
        Fnew += replace(F, {t: tnew, u: unew, v: vnew})
            
    return Fnew, k


# TODO: get the form where we get a dense mass but block diagonal
# stiffness part.  That's just a simple bit.
def getFormW(F, butcher, t, dt, u0):
    pass



if __name__ == "__main__":
    BT = butcher.LobattoIIIAButcherTableau(2)

    ns = len(BT.b)
    N = 4
    msh = UnitSquareMesh(N, N)

    V = FunctionSpace(msh, "CG", 1)
    x, y = SpatialCoordinate(msh)

    # Stores initial condition
    u = interpolate(2.0 + sin(pi*x)*sin(pi*y), V)
    v = TestFunction(V)

    F = inner(grad(u), grad(v))*dx

    dt = Constant(0.1)
    t = Constant(0.0)

    Fnew, k = getForm(F, BT, t, dt, u) 

    params = {"mat_type": "aij",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    prob = NonlinearVariationalProblem(Fnew, k)
    solver = NonlinearVariationalSolver(prob, solver_parameters=params)
    print(assemble(u**2*dx))
    solver.solve()

    unew = u.copy(deepcopy=True)

    for i in range(ns):
        unew.dat.data[:] += dt.values()[0] * BT.b[i] * k.dat[i].data

    # And we've got damping!
    print(assemble(unew**2*dx))
