import numpy
from firedrake import (TestFunction, Function, Constant,
                       VectorFunctionSpace,
                       as_vector,
                       split, DirichletBC, interpolate, project)
from firedrake.dmhooks import push_parent
from ufl import replace, diff
from ufl.algorithms import expand_derivatives
from ufl.classes import Zero
from .deriv import TimeDerivative  # , apply_time_derivatives


def getForm(F, butch, t, dt, u0, bcs=None):
    """Given a time-dependent variational form and a
    :class:`ButcherTableau`, produce UFL for the s-stage RK method.

    :arg F: UFL form for the semidiscrete ODE/DAE
    :arg butch: the :class:`ButcherTableau` for the RK method being used to
         advance in time.
    :arg t: a :class:`Constant` referring to the current time level.
         Any explicit time-dependence in F is included
    :arg dt: a :class:`Constant` referring to the size of the current
         time step.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
         containing (possible time-dependent) boundary conditions imposed
         on the system.

    On output, we return a tuple consisting of four parts:

       - Fnew, the :class:`Form`
       - k, the :class:`firedrake.Function` holding all the stages.
         It lives in a :class:`firedrake.FunctionSpace` corresponding to the
         s-way tensor product of the space on which the semidiscrete
         form lives.
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the stages,
       - `gblah`, a list of pairs of the form (f, expr), where f is
         a :class:`firedrake.Function` and expr is a :class:`ufl.Expr`.
         at each time step, each expr needs to be re-interpolated/projected
         onto the corresponding f in order for Firedrake to pick up that
         time-dependent boundary conditions need to be re-applied.
"""

    v = F.arguments()[0]
    V = v.function_space()
    assert V == u0.function_space()

    A = numpy.array([[Constant(aa) for aa in arow] for arow in butch.A])
    c = numpy.array([Constant(ci) for ci in butch.c])

    num_stages = butch.num_stages
    num_fields = len(V)

    if V.value_size == 1 and num_stages > 1:
        Vbig = VectorFunctionSpace(V.mesh(), V.ufl_element(), dim=num_stages)
    else:
        Vbig = numpy.prod([V for i in range(num_stages)])

    # Silence a warning about transfer managers when we
    # coarsen coefficients in V
    push_parent(V.dm, Vbig.dm)
    vnew = TestFunction(Vbig)
    k = Function(Vbig)
    vbits = split(v)
    vbigbits = split(vnew)
    u0bits = split(u0)
    kbits = split(k)

    Ak = A @ numpy.reshape(kbits, (num_stages, num_fields))


    Fnew = Zero()

    for i in range(num_stages):
        repl = {t: t + c[i] * dt}
        for j, (ubit, vbit, kbit) in enumerate(zip(u0bits, vbits, kbits)):
            repl[ubit] = ubit + dt * Ak[i, j]
            repl[vbit] = vbigbits[num_fields * i + j]
            repl[TimeDerivative(ubit)] = kbits[num_fields * i + j]

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
        if V.value_size == 1:
            curbc_stuff = []
            for i in range(num_stages):
                gcur = replace(gfoo, {t: t + c[i] * dt})
                try:
                    gdat = interpolate(gcur, V)
                except NotImplementedError:
                    gdat = project(gcur, V)
                gblah.append((gdat, gcur))
                curbc_stuff.append(gdat)
            bcnew.append(DirichletBC(Vbig, as_vector(curbc_stuff), boundary))
        elif len(V) == 1:
            for i in range(num_stages):
                gcur = replace(gfoo, {t: t + c[i] * dt})
                try:
                    gdat = interpolate(gcur, V)
                except NotImplementedError:
                    gdat = project(gcur, V)
                gblah.append((gdat, gcur))
                bcnew.append(DirichletBC(Vbig[i], gdat, boundary))
        else:
            sub = bc.function_space_index()
            for i in range(num_stages):
                gcur = replace(gfoo, {t: t + butch.c[i] * dt})
                try:
                    gdat = interpolate(gcur, V)
                except NotImplementedError:
                    gdat = project(gcur, V)
                gblah.append((gdat, gcur))
                bcnew.append(DirichletBC(Vbig[sub + num_fields * i],
                                         gdat, boundary))

    return Fnew, k, bcnew, gblah
