import numpy
from firedrake import TestFunction, Function, inner, Constant, dx, split
from ufl import replace


def getForm(F, butch, t, dt, u0, bcs=None):
    """Given a variational form F(u; v) describing a nonlinear
    time-dependent problem (u_t, v, t) + F(u, t; v) and a
    Butcher tableau, produce UFL for the s-stage RK method.

    :arg F: UFL form for the ODE part (leaving off the time derivative)
    :arg butch: the Butcher tableu for the RK method being used to
         advance in time.
    :arg t: a :class:`Constant` referring to the current time level.
         Any explicit time-dependence in F is included here
    :arg dt: a :class:`Constant` referring to the size of the current
         time step.
    :arg u0: a :class:`Function` referring to the current state of
         the PDE system
    :arg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
         containing (possible time-dependent) boundary conditions imposed
         on the system.  CURRENTLY ASSUMED to be None

    On output, we return UFL for a single time-step of the RK method and a
    handle on the :class:`Function` on the product function space that is
    used to store the RK stages.  This would be the starting function/solution
    to a :class:`NonlinearVariationalProblem` for all the stages.
"""
    if bcs is not None:
        raise NotImplementedError("Don't have BCs worked out yet")
    
    v, = F.arguments()

    V = v.function_space()

    assert V == u0.function_space()


    A = butch.A
    c = butch.c

    num_stages = len(c)
    
    if len(V) == 1 and num_stages == 1:
        # don't need a new test function in this case.
        k = Function(V)
        Fnew = inner(k, v)*dx
        tnew = t + Constant(c[0]) * dt
        unew = u0 + dt * Constant(A[0, 0]) * k
        Fnew += replace(F, {t: tnew, u0: unew})
    elif len(V) == 1 and num_stages > 1:  # multi-stage method
        Vbig = numpy.prod([V for i in range(num_stages)])
        vnew = TestFunction(Vbig)
        k = Function(Vbig)
        
        Fnew = inner(k, vnew)*dx
        for i in range(num_stages):
            unew = u0 + dt * numpy.sum([Constant(A[i, j]) * k[j]
                                        for j in range(num_stages)])
            tnew = t + Constant(c[i]) * dt
            Fnew += replace(F, {t: tnew,
                                u0: unew,
                                test: vnew[i]})
    elif len(V) > 1 and num_stages == 1:
        k = Function(V)
        Fnew = inner(k, v)*dx
        tnew = t + Constant(c[0]) * dt
        ubits = split(u0)
        kbits = split(k)
        repl = {ubits[i]: ubits[i] + dt * Constant(A[0, 0]) * kbits[i] \
                for i in range(len(V))}
        Fnew += replace(F, repl)
    elif len(V) > 1 and num_stages > 1:
        Vbig = numpy.prod([V for i in range(num_stages)])
        vnew = TestFunction(Vbig)
        Fnew = inner(k, v)*dx
        1/0  # still working on it!


    return Fnew, k


def getFormW(F, butch, t, dt, u0):
    """When the Butcher matrix butch.A is invertible, it is possible
    to reformulate the variational problem to make the mass part of
    the Jacobian denser but the stiffness part block diagonal.  This
    can make certain kinds of block preconditioners far more
    effective, and assembly of the Jacobian cheaper as well."""

    raise NotImplementedError()
