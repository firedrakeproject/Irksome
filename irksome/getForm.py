from functools import reduce
from operator import mul

import numpy
from firedrake import Function, TestFunction, split
from ufl import diff
from ufl.algorithms import expand_derivatives
from ufl.classes import Zero
from ufl.constantvalue import as_ufl
from .tools import ConstantOrZero, MeshConstant, replace, getNullspace, AI
from .deriv import TimeDerivative  # , apply_time_derivatives
from .bcs import BCStageData, bc2space, stage2spaces4bc


def getForm(F, butch, t, dt, u0, bcs=None, bc_type=None, splitting=AI,
            nullspace=None):
    """Given a time-dependent variational form and a
    :class:`ButcherTableau`, produce UFL for the s-stage RK method.

    :arg F: UFL form for the semidiscrete ODE/DAE
    :arg butch: the :class:`ButcherTableau` for the RK method being used to
         advance in time.
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
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
    """
    if bc_type is None:
        bc_type = "DAE"
    v = F.arguments()[0]
    V = v.function_space()
    msh = V.mesh()
    assert V == u0.function_space()

    MC = MeshConstant(msh)

    c = numpy.array([MC.Constant(ci) for ci in butch.c],
                    dtype=object)

    bA1, bA2 = splitting(butch.A)

    try:
        bA1inv = numpy.linalg.inv(bA1)
    except numpy.linalg.LinAlgError:
        bA1inv = None
    try:
        bA2inv = numpy.linalg.inv(bA2)
        A2inv = numpy.array([[ConstantOrZero(aa, MC) for aa in arow] for arow in bA2inv],
                            dtype=object)
    except numpy.linalg.LinAlgError:
        raise NotImplementedError("We require A = A1 A2 with A2 invertible")

    A1 = numpy.array([[ConstantOrZero(aa, MC) for aa in arow] for arow in bA1],
                     dtype=object)
    if bA1inv is not None:
        A1inv = numpy.array([[ConstantOrZero(aa, MC) for aa in arow] for arow in bA1inv],
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

    if bcs is None:
        bcs = []
    if bc_type == "ODE":
        assert splitting == AI, "ODE-type BC aren't implemented for this splitting strategy"
        u0_mult_np = numpy.divide(1.0, butch.c, out=numpy.zeros_like(butch.c), where=butch.c != 0)
        u0_mult = numpy.array([MC.Constant(0) for mi in u0_mult_np],
                              dtype=object)

        def bc2gcur(bc, i):
            gorig = as_ufl(bc._original_arg)
            gfoo = expand_derivatives(diff(gorig, t))
            return replace(gfoo, {t: t + c[i] * dt}) + u0_mult[i]*gorig

    elif bc_type == "DAE":
        if bA1inv is None:
            raise NotImplementedError("Cannot have DAE BCs for this Butcher Tableau/splitting")

        u0_mult_np = A1inv @ numpy.ones_like(butch.c)
        u0_mult = numpy.array([ConstantOrZero(mi, MC)/dt for mi in u0_mult_np],
                              dtype=object)

        def bc2gcur(bc, i):
            gorig = as_ufl(bc._original_arg)
            gcur = 0
            for j in range(num_stages):
                gcur += ConstantOrZero(bA1inv[i, j], MC) / dt * replace(gorig, {t: t + c[j]*dt})
            return gcur
    else:
        raise ValueError("Unrecognised bc_type: %s", bc_type)

    # This logic uses information set up in the previous section to
    # set up the new BCs for either method
    for bc in bcs:
        for i in range(num_stages):
            Vsp = bc2space(bc, V)
            Vbigi = stage2spaces4bc(bc, V, Vbig, i)
            gcur = bc2gcur(bc, i)
            gdat = BCStageData(Vsp, gcur, u0, u0_mult, i, t, dt)
            bcnew.append(bc.reconstruct(V=Vbigi, g=gdat))

    nspnew = getNullspace(V, Vbig, butch, nullspace)

    return Fnew, w, bcnew, nspnew
