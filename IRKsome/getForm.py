import numpy
from firedrake import TestFunction, Function, inner, Constant, dx, split, DirichletBC, interpolate
from ufl import replace, diff
from ufl.algorithms import expand_derivatives


class RK(object):
    def __init__(F, butch, t, dt, u0, bcs=None):
        self.BT = butch
        self.F = F
        self.t = t
        self.dt = dt
        self.u0 = u0
        self.bcs = bcs
        self.bigF = None
        
    def advance(self):
        pass

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

    v = F.arguments()[0]
    V = v.function_space()
    assert V == u0.function_space()

    A = numpy.array([[Constant(aa) for aa in arow] for arow in butch.A])
    c = numpy.array([Constant(ci) for ci in butch.c])

    num_stages = len(c)
    num_fields = len(V)

    Vbig = numpy.prod([V for i in range(num_stages)])
    vnew = TestFunction(Vbig)
    k = Function(Vbig)
    vbits = split(v)
    vbigbits = split(vnew)
    u0bits = split(u0)
    kbits = split(k)

    Ak = A @ numpy.reshape(kbits, (num_stages, num_fields))
    Fnew = inner(k, vnew) * dx

    for i in range(num_stages):
        repl = {t: t + c[i] * dt}
        for j, (ubit, vbit) in enumerate(zip(u0bits, vbits)):
            repl[ubit] = ubit + dt * Ak[i, j]
            repl[vbit] = vbigbits[num_fields * i + j]

        Fnew += replace(F, repl)

    bcnew = []
    gblah = []

    if bcs is None:
        bcs = []
    for bc in bcs:
        if bc.domain_args[0] == "on_boundary":
            boundary = "on_boundary"
        else:
            boundary = ()
            for j in bc.domain_args[1][1]:
                boundary += j 
        gfoo = expand_derivatives(diff(bc._original_val, t))
        print(gfoo)
        for i in range(num_stages):
            gcur = replace(gfoo, {t: t+Constant(butch.c[i])*dt})
            gdat = interpolate(gcur, V)

            gblah.append((gdat, gcur))

            bcnew.append(DirichletBC(Vbig[i], gdat, "on_boundary"))

    return Fnew, k, (bcnew, gblah)


def getFormW(F, butch, t, dt, u0):
    """When the Butcher matrix butch.A is invertible, it is possible
    to reformulate the variational problem to make the mass part of
    the Jacobian denser but the stiffness part block diagonal.  This
    can make certain kinds of block preconditioners far more
    effective, and assembly of the Jacobian cheaper as well."""

    raise NotImplementedError()
