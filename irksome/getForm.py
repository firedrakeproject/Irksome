from functools import reduce
from operator import mul

import numpy
from firedrake import (Constant, DirichletBC, Function, TestFunction,
                       interpolate, project, split)
from ufl import diff
from ufl.algorithms import expand_derivatives
from ufl.classes import Zero
from ufl.constantvalue import as_ufl
from .tools import replace, getNullspace, AI
from .deriv import TimeDerivative  # , apply_time_derivatives


class BCStageData(object):
    def __init__(self, V, gcur, u0, u0_mult, i, t, dt):
        if V.component is not None:     # bottommost space is bit of VFS
            if V.parent.index is None:  # but not part of a MFS
                sub = V.component
                try:
                    gdat = interpolate(gcur-u0_mult[i]*u0.sub(sub), V)
                    gmethod = lambda g, u: gdat.interpolate(g-u0_mult[i]*u.sub(sub))
                except:  # noqa: E722
                    gdat = project(gcur-u0_mult[i]*u0.sub(sub), V)
                    gmethod = lambda g, u: gdat.project(g-u0_mult[i]*u.sub(sub))
            else:   # V is a bit of a VFS inside an MFS
                sub0 = V.parent.index
                sub1 = V.component
                try:
                    gdat = interpolate(gcur-u0_mult[i]*u0.sub(sub0).sub(sub1), V)
                    gmethod = lambda g, u: gdat.interpolate(g-u0_mult[i]*u.sub(sub0).sub(sub1))
                except:  # noqa: E722
                    gdat = project(gcur-u0_mult[i]*u0.sub(sub0).sub(sub1), V)
                    gmethod = lambda g, u: gdat.project(g-u0_mult[i]*u.sub(sub0).sub(sub1))
        else:  # V is not a bit of a VFS
            if V.index is None:  # not part of MFS, either
                try:
                    gdat = interpolate(gcur-u0_mult[i]*u0, V)
                    gmethod = lambda g, u: gdat.interpolate(g-u0_mult[i]*u)
                except:  # noqa: E722
                    gdat = project(gcur-u0_mult[i]*u0, V)
                    gmethod = lambda g, u: gdat.project(g-u0_mult[i]*u)
            else:  # part of MFS
                sub = V.index
                try:
                    gdat = interpolate(gcur-u0_mult[i]*u0.sub(sub), V)
                    gmethod = lambda g, u: gdat.interpolate(g-u0_mult[i]*u.sub(sub))
                except:  # noqa: E722
                    gdat = project(gcur-u0_mult[i]*u0.sub(sub), V)
                    gmethod = lambda g, u: gdat.project(g-u0_mult[i]*u.sub(sub))

        self.gstuff = (gdat, gcur, gmethod)


def ConstantOrZero(x):
    return Zero() if abs(complex(x)) < 1.e-10 else Constant(x)


def getForm(F, butch, t, dt, u0, bcs=None, bc_type=None, splitting=AI,
            nullspace=None):
    """Given a time-dependent variational form and a
    :class:`ButcherTableau`, produce UFL for the s-stage RK method.

    :arg F: UFL form for the semidiscrete ODE/DAE
    :arg butch: the :class:`ButcherTableau` for the RK method being used to
         advance in time.
    :arg t: a :class:`Constant` referring to the current time level.
         Any explicit time-dependence in F is included
    :arg dt: a :class:`Constant` referring to the size of the current
         time step.
    :arg splitting: a callable that maps the (floating point) Butcher matrix
         a to a pair of matrices `A1, A2` such that `butch.A = A1 A2`.  This is used
         to vary between the classical RK formulation and Butcher's reformulation
         that leads to a denser mass matrix with block-diagonal stiffness.
         Some choices of function will assume that `butch.A` is invertible.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
         containing (possible time-dependent) boundary conditions imposed
         on the system.
    :arg bc_type: How to manipulate the strongly-enforced boundary
         conditions to derive the stage boundary conditions.  Should
         be a string, either "DAE", which implements BCs as
         constraints in the style of a differential-algebraic
         equation, or "ODE", which takes the time derivative of the
         boundary data and evaluates this for the stage values
    :arg nullspace: A list of tuples of the form (index, VSB) where
         index is an index into the function space associated with `u`
         and VSB is a :class: `firedrake.VectorSpaceBasis` instance to
         be passed to a `firedrake.MixedVectorSpaceBasis` over the
         larger space associated with the Runge-Kutta method

    On output, we return a tuple consisting of four parts:

       - Fnew, the :class:`Form`
       - k, the :class:`firedrake.Function` holding all the stages.
         It lives in a :class:`firedrake.FunctionSpace` corresponding to the
         s-way tensor product of the space on which the semidiscrete
         form lives.
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the stages,
       - 'nspnew', the :class:`firedrake.MixedVectorSpaceBasis` object
         that represents the nullspace of the coupled system
       - `gblah`, a list of tuples of the form (f, expr, method),
         where f is a :class:`firedrake.Function` and expr is a
         :class:`ufl.Expr`.  At each time step, each expr needs to be
         re-interpolated/projected onto the corresponding f in order
         for Firedrake to pick up that time-dependent boundary
         conditions need to be re-applied.  The
         interpolation/projection is encoded in method, which is
         either `f.interpolate(expr-c*u0)` or `f.project(expr-c*u0)`, depending
         on whether the function space for f supports interpolation or
         not.
    """
    if bc_type is None:
        bc_type = "DAE"
    v = F.arguments()[0]
    V = v.function_space()
    assert V == u0.function_space()

    c = numpy.array([Constant(ci) for ci in butch.c],
                    dtype=object)

    bA1, bA2 = splitting(butch.A)

    try:
        bA1inv = numpy.linalg.inv(bA1)
    except numpy.linalg.LinAlgError:
        bA1inv = None
    try:
        bA2inv = numpy.linalg.inv(bA2)
        A2inv = numpy.array([[ConstantOrZero(aa) for aa in arow] for arow in bA2inv],
                            dtype=object)
    except numpy.linalg.LinAlgError:
        raise NotImplementedError("We require A = A1 A2 with A2 invertible")

    A1 = numpy.array([[ConstantOrZero(aa) for aa in arow] for arow in bA1],
                     dtype=object)
    if bA1inv is not None:
        A1inv = numpy.array([[ConstantOrZero(aa) for aa in arow] for arow in bA1inv],
                            dtype=object)
    else:
        A1inv = None

    num_stages = butch.num_stages
    num_fields = len(V)

    Vbig = reduce(mul, (V for _ in range(num_stages)))

    vnew = TestFunction(Vbig)
    w = Function(Vbig)

    if len(V) == 1:
        u0bits = [u0]
        vbits = [v]
        if num_stages == 1:
            vbigbits = [vnew]
            wbits = [w]
        else:
            vbigbits = split(vnew)
            wbits = split(w)
    else:
        u0bits = split(u0)
        vbits = split(v)
        vbigbits = split(vnew)
        wbits = split(w)

    wbits_np = numpy.zeros((num_stages, num_fields), dtype=object)

    for i in range(num_stages):
        for j in range(num_fields):
            wbits_np[i, j] = wbits[i*num_fields+j]

    A1w = A1 @ wbits_np
    A2invw = A2inv @ wbits_np

    Fnew = Zero()

    for i in range(num_stages):
        repl = {t: t + c[i] * dt}
        for j, (ubit, vbit) in enumerate(zip(u0bits, vbits)):
            repl[ubit] = ubit + dt * A1w[i, j]
            repl[vbit] = vbigbits[num_fields * i + j]
            repl[TimeDerivative(ubit)] = A2invw[i, j]
            if (len(ubit.ufl_shape) == 1):
                for kk in range(len(A1w[i, j])):
                    repl[TimeDerivative(ubit[kk])] = A2invw[i, j][kk]
                    repl[ubit[kk]] = repl[ubit][kk]
                    repl[vbit[kk]] = repl[vbit][kk]
        Fnew += replace(F, repl)

    bcnew = []
    gblah = []

    if bcs is None:
        bcs = []
    if bc_type == "ODE":
        assert splitting == AI, "ODE-type BC aren't implemented for this splitting strategy"
        u0_mult_np = numpy.divide(1.0, butch.c, out=numpy.zeros_like(butch.c), where=butch.c != 0)
        u0_mult = numpy.array([ConstantOrZero(mi)/dt for mi in u0_mult_np],
                              dtype=object)

        def bc2gcur(bc, i):
            gorig = bc._original_arg
            gfoo = expand_derivatives(diff(gorig, t))
            return replace(gfoo, {t: t + c[i] * dt}) + u0_mult[i]*gorig

    elif bc_type == "DAE":
        if bA1inv is None:
            raise NotImplementedError("Cannot have DAE BCs for this Butcher Tableau/splitting")

        u0_mult_np = A1inv @ numpy.ones_like(butch.c)
        u0_mult = numpy.array([ConstantOrZero(mi)/dt for mi in u0_mult_np],
                              dtype=object)

        def bc2gcur(bc, i):
            gorig = as_ufl(bc._original_arg)
            gcur = 0
            for j in range(num_stages):
                gcur += ConstantOrZero(bA1inv[i, j]) / dt * replace(gorig, {t: t + c[j]*dt})
            return gcur
    else:
        raise ValueError("Unrecognised bc_type: %s", bc_type)

    # This logic uses information set up in the previous section to
    # set up the new BCs for either method
    for bc in bcs:
        if num_fields == 1:  # not mixed space
            comp = bc.function_space().component
            if comp is not None:  # check for sub-piece of vector-valued
                Vsp = V.sub(comp)
                Vbigi = lambda i: Vbig[i].sub(comp)
            else:
                Vsp = V
                Vbigi = lambda i: Vbig[i]
        else:  # mixed space
            sub = bc.function_space_index()
            comp = bc.function_space().component
            if comp is not None:  # check for sub-piece of vector-valued
                Vsp = V.sub(sub).sub(comp)
                Vbigi = lambda i: Vbig[sub+num_fields*i].sub(comp)
            else:
                Vsp = V.sub(sub)
                Vbigi = lambda i: Vbig[sub+num_fields*i]

        for i in range(num_stages):
            gcur = bc2gcur(bc, i)
            blah = BCStageData(Vsp, gcur, u0, u0_mult, i, t, dt)
            gdat, gcr, gmethod = blah.gstuff
            gblah.append((gdat, gcr, gmethod))
            bcnew.append(DirichletBC(Vbigi(i), gdat, bc.sub_domain))

    nspnew = getNullspace(V, Vbig, butch, nullspace)

    return Fnew, w, bcnew, nspnew, gblah
