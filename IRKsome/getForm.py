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
    
    v = F.arguments()[0]

    V = v.function_space()

    assert V == u0.function_space()

    A = numpy.array([[Constant(aa) for aa in arow] for arow in butch.A])
    c = butch.c

    num_stages = len(c)
    num_fields = len(V)
    
    if num_fields == 1 and num_stages == 1:
        # don't need a new test function in this case.
        k = Function(V)
        Fnew = inner(k, v)*dx
        tnew = t + Constant(c[0]) * dt
        unew = u0 + dt * A[0, 0] * k
        Fnew += replace(F, {t: tnew, u0: unew})
    elif num_fields == 1 and num_stages > 1:  # multi-stage method
        Vbig = numpy.prod([V for i in range(num_stages)])
        vnew = TestFunction(Vbig)
        k = Function(Vbig)
        Fnew = inner(k, vnew)*dx
        Ak = A @ k
        for i in range(num_stages):
            unew = u0 + dt * Ak[i]
            tnew = t + Constant(c[i]) * dt
            Fnew += replace(F, {t: tnew,
                                u0: unew,
                                v: vnew[i]})
    elif num_fields > 1 and num_stages == 1:
        k = Function(V)
        kbits = split(k)
        u0bits = split(u0)
        repl = {t: t + Constant(c[0]) * dt}
        for ubit, kbit in zip(u0bits, kbits):
            repl[ubit] = ubit + dt * A[0, 0] * kbit
        Fnew = inner(k, v)*dx + replace(F, repl)
    elif num_fields > 1 and num_stages > 1:
        Vbig = numpy.prod([V for i in range(num_stages)])
        vnew = TestFunction(Vbig)
        k = Function(Vbig)
        1/0

    return Fnew, k


def getFormW(F, butch, t, dt, u0):
    """When the Butcher matrix butch.A is invertible, it is possible
    to reformulate the variational problem to make the mass part of
    the Jacobian denser but the stiffness part block diagonal.  This
    can make certain kinds of block preconditioners far more
    effective, and assembly of the Jacobian cheaper as well."""

    raise NotImplementedError()
