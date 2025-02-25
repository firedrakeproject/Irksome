from functools import reduce
from operator import mul

import numpy
from firedrake import Constant, Function, TestFunction
from ufl import as_tensor, diff, dot
from ufl.algorithms import expand_derivatives
from ufl.classes import Zero
from ufl.constantvalue import as_ufl
from .tools import replace, getNullspace, AI, numpy_to_ufl
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
         containing (possibly time-dependent) boundary conditions imposed
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
    assert V == u0.function_space()

    c = Constant(butch.c)

    bA1, bA2 = splitting(butch.A)

    try:
        bA1inv = numpy.linalg.inv(bA1)
    except numpy.linalg.LinAlgError:
        bA1inv = None
    try:
        bA2inv = numpy.linalg.inv(bA2)
    except numpy.linalg.LinAlgError:
        raise NotImplementedError("We require A = A1 A2 with A2 invertible")

    if bA1inv is not None:
        A1inv = Constant(bA1inv)
    else:
        A1inv = None

    num_stages = butch.num_stages
    Vbig = reduce(mul, (V for _ in range(num_stages)))

    w = Function(Vbig)
    vnew = TestFunction(Vbig)
    vflat = numpy.reshape(vnew, (num_stages, *u0.ufl_shape))
    wflat = numpy.reshape(w, (num_stages, *u0.ufl_shape))

    A1 = numpy_to_ufl(bA1)
    A2inv = numpy_to_ufl(bA2inv)

    A1w = dot(A1, as_tensor(wflat))
    A2invw = dot(A2inv, as_tensor(wflat))

    Fnew = Zero()
    dtu = TimeDerivative(u0)
    for i in range(num_stages):
        ii = (i,) + (slice(None),) * (len(A1w.ufl_shape)-1)
        repl = {t: t + c[i] * dt}

        # Replace entire mixed function
        repl[v] = as_tensor(vflat[i])
        repl[u0] = u0 + dt * A1w[ii]
        repl[dtu] = A2invw[ii]

        if u0.ufl_shape:
            for kk in numpy.ndindex(u0.ufl_shape):
                # Replace each scalar component
                repl[v[kk]] = repl[v][kk]
                repl[u0[kk]] = repl[u0][kk]
                repl[TimeDerivative(u0[kk])] = repl[dtu][kk]

        Fnew += replace(F, repl)

    bcnew = []

    if bcs is None:
        bcs = []
    if bc_type == "ODE":
        assert splitting == AI, "ODE-type BC aren't implemented for this splitting strategy"
        u0_mult = Zero(butch.c.shape)

        def bc2gcur(bc, i):
            gorig = as_ufl(bc._original_arg)
            gfoo = expand_derivatives(diff(gorig, t))
            return replace(gfoo, {t: t + c[i] * dt}) + u0_mult[i]*gorig

    elif bc_type == "DAE":
        if bA1inv is None:
            raise NotImplementedError("Cannot have DAE BCs for this Butcher Tableau/splitting")

        u0_mult = dot(A1inv, as_tensor(numpy.ones_like(butch.c))) / dt

        def bc2gcur(bc, i):
            gorig = as_ufl(bc._original_arg)
            gcur = (1/dt)*sum(A1inv[i, j] * replace(gorig, {t: t + c[j]*dt}) for j in range(num_stages))
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

    nspnew = getNullspace(V, Vbig, num_stages, nullspace)

    return Fnew, w, bcnew, nspnew
