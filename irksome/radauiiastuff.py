# Applies RadauIIA methods (very special case!) in terms of the polynomial
# values at each interpolation point instead of in terms
# of the RK stages.  This makes BCs and additively partitioned methods
# easier to implement
# No DAE for now, just ODE-type problems.


from functools import reduce
from operator import mul

import numpy
from firedrake import (Constant, Function, TestFunction, dx, inner,
                       interpolate, project, split, MixedVectorSpaceBasis)
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake.dmhooks import push_parent
# from ufl import diff
from ufl.algorithms import expand_derivatives
from ufl.algorithms.analysis import has_exact_type
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import CoefficientDerivative, Zero
from ufl.constantvalue import as_ufl
from ufl.corealg.multifunction import MultiFunction
from ufl.log import error

from .deriv import TimeDerivative  # , apply_time_derivatives


# post-order traversal instead of pre-order used in UFL
class MyReplacer(MultiFunction):
    def __init__(self, mapping):
        super().__init__()
        self.replacements = mapping
        if not all(k.ufl_shape == v.ufl_shape for k, v in mapping.items()):
            error("Replacement expressions must have the same shape as what they replace.")

    def expr(self, o):
        if o in self.replacements:
            return self.replacements[o]
        else:
            return self.reuse_if_untouched(o, *map(self, o.ufl_operands))


def replace(e, mapping):
    """Replace subexpressions in expression.

    @param e:
        An Expr or Form.
    @param mapping:
        A dict with from:to replacements to perform.
    """
    mapping2 = dict((k, as_ufl(v)) for (k, v) in mapping.items())

    # Workaround for problem with delayed derivative evaluation
    # The problem is that J = derivative(f(g, h), g) does not evaluate immediately
    # So if we subsequently do replace(J, {g: h}) we end up with an expression:
    # derivative(f(h, h), h)
    # rather than what were were probably thinking of:
    # replace(derivative(f(g, h), g), {g: h})
    #
    # To fix this would require one to expand derivatives early (which
    # is not attractive), or make replace lazy too.
    if has_exact_type(e, CoefficientDerivative):
        # Hack to avoid circular dependencies
        e = expand_derivatives(e)

    return map_integrand_dags(MyReplacer(mapping2), e)


def ConstantOrZero(x):
    return Zero() if abs(complex(x)) < 1.e-10 else Constant(x)


# FIXME: can we streamline this for RadauIIA/collocation?
class BCStageData(object):
    def __init__(self, V, gcur, u0, u0_mult, i, t, dt):
        if V.index is None:  # Not part of a mixed space
            try:
                gdat = interpolate(gcur-u0_mult[i]*u0, V)
                gmethod = lambda g, u: gdat.interpolate(g-u0_mult[i]*u)
            except:  # noqa: E722
                gdat = project(gcur-u0_mult[i]*u0, V)
                gmethod = lambda g, u: gdat.project(g-u0_mult[i]*u)
        else:
            sub = V.index
            try:
                gdat = interpolate(gcur-u0_mult[i]*u0.sub(sub), V)
                gmethod = lambda g, u: gdat.interpolate(g-u0_mult[i]*u.sub(sub))
            except:  # noqa: E722
                gdat = project(gcur-u0_mult[i]*u0.sub(sub), V)
                gmethod = lambda g, u: gdat.project(g-u0_mult[i]*u.sub(sub))

        self.gstuff = (gdat, gcur, gmethod)


def getForm(F, butch, t, dt, u0, bcs=None, nullspace=None):
    """Given a time-dependent variational form and a
    :class:`RadauIIA`, produce UFL for the s-stage RK method.
         in terms of the point values rather than stage values.
    :arg F: UFL form for the (u_t, v) + F(t, u; v) = 0
    :arg butch: the :class:`RadauIIA` the collocation tableau being used
    :arg t: a :class:`Constant` referring to the current time level.
         Any explicit time-dependence in F is included through t.
    :arg dt: a :class:`Constant` referring to the size of the current
         time step.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
         containing (possible time-dependent) boundary conditions imposed
         on the system.
    :arg nullspace: A list of tuples of the form (index, VSB) where
         index is an index into the function space associated with `u`
         and VSB is a :class: `firedrake.VectorSpaceBasis` instance to
         be passed to a `firedrake.MixedVectorSpaceBasis` over the
         larger space associated with the Runge-Kutta method

    On output, we return a tuple consisting of four parts:

       - Fnew, the variational :class:`Form` for the problem we solve
         at each time step
       - UU, the :class:`firedrake.Function` for storing the solution at the 
         collocation times.
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
         conditions need to be re-applied.
         not.
    """
    assert bcs is None and nullspace is None

    v = F.arguments()[0]
    V = v.function_space()
    assert V == u0.function_space()

    c = numpy.array([Constant(ci) for ci in butch.c],
                    dtype=object)

    A1 = numpy.array([[ConstantOrZero(aa) for aa in arow]
                      for arow in butch.A], dtype=object)

    num_stages = butch.num_stages
    num_fields = len(V)

    Vbig = reduce(mul, (V for _ in range(num_stages)))
    # Silence a warning about transfer managers when we
    # coarsen coefficients in V
    push_parent(V.dm, Vbig.dm)

    vnew = TestFunction(Vbig)
    UU = Function(Vbig)

    if len(V) == 1:
        u0bits = [u0]
        vbits = [v]
        if num_stages == 1:
            vbigbits = [vnew]
            UUbits = [UU]
        else:
            vbigbits = split(vnew)
            UUbits = split(UU)
    else:
        u0bits = split(u0)
        vbits = split(v)
        vbigbits = split(vnew)
        UUbits = split(UU)

    UUbits = numpy.reshape(
        numpy.asarray(UUbits), (num_stages, num_fields))
    vbigbits = numpy.reshape(
        numpy.asarray(vbigbits), (num_stages, num_fields))

    Fnew = Zero()
    for i in range(num_stages):
        for j in range(num_fields):
            Fnew += inner(UUbits[i, j] - u0bits[j], vbigbits[i, j]) * dx

    # Now substitute into F for each stage, we need a double-loop over
    # the stages for this.
    for i0 in range(num_stages):
        repl = {t: t + c[i0] * dt}

        # test functions within the split
        for f in range(num_fields):
            repl[vbits[f]] = vbigbits[i0, f]

        for i1 in range(num_stages):
            for f in range(num_fields):
                repl[u0bits[f]] = UUbits[i1, f]

            Fnew += dt * A1[i0, i1] * replace(F, repl)

    if nullspace is None:
        nspnew = None
    else:
        try:
            nullspace.sort()
        except TypeError:
            raise TypeError("Nullspace entries must be of form (idx, VSP), where idx is a non-negative integer")
        if (nullspace[-1][0] > num_fields) or (nullspace[0][0] < 0):
            raise ValueError("At least one index for nullspaces is out of range")
        nspnew = []
        for i in range(num_stages):
            count = 0
            for j in range(num_fields):
                if j == nullspace[count][0]:
                    nspnew.append(nullspace[count][1])
                    count += 1
                else:
                    nspnew.append(Vbig.sub(j + num_fields * i))
        nspnew = MixedVectorSpaceBasis(Vbig, nspnew)

    return Fnew, UU, None, nspnew, None


class TimeStepper:
    """Front-end class for advancing a time-dependent PDE via a Runge-Kutta
    method.

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
            the Runge-Kutta method to be used for time marching.
    :arg t: A :class:`firedrake.Constant` instance that always
            contains the time value at the beginning of a time step
    :arg dt: A :class:`firedrake.Constant` containing the size of the
            current time step.  The user may adjust this value between
            time steps, but see :class:`AdaptiveTimeStepper` for a
            method that attempts to do this automatically.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing
            the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the RK method.
    :arg bc_type: How to manipulate the strongly-enforced boundary
            conditions to derive the stage boundary conditions.
            Should be a string, either "DAE", which implements BCs as
            constraints in the style of a differential-algebraic
            equation, or "ODE", which takes the time derivative of the
            boundary data and evaluates this for the stage values
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    :arg nullspace: A list of tuples of the form (index, VSB) where
            index is an index into the function space associated with
            `u` and VSB is a :class: `firedrake.VectorSpaceBasis`
            instance to be passed to a
            `firedrake.MixedVectorSpaceBasis` over the larger space
            associated with the Runge-Kutta method
    """
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None, nullspace=None):
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = len(butcher_tableau.b)
        self.butcher_tableau = butcher_tableau

        bigF, U, bigBCs, bigNSP, bigBCdata = \
            getForm(F, butcher_tableau, t, dt, u0, bcs, nullspace)

        self.U = U
        self.Us = U.split()
        self.bigBCs = bigBCs
        self.bigBCdata = bigBCdata
        problem = NLVP(bigF, self.U, bigBCs)

        appctx = {"F": F,
                  "butcher_tableau": butcher_tableau,
                  "t": t,
                  "dt": dt,
                  "u0": u0,
                  "bcs": bcs,
                  "nullspace": nullspace}
        self.solver = NLVS(problem,
                           appctx=appctx,
                           solver_parameters=solver_parameters,
                           nullspace=bigNSP)

    def advance(self):
        """Advances the system from time `t` to time `t + dt`.
        Note: overwrites the value `u0`."""

        # Diddle with BC...

        self.solver.solve()

        # the last shall be first.
        ns = self.num_stages
        nf = self.num_fields
        for i, u0d in enumerate(self.u0.dat):
            u0d.data[:] = self.Us[nf*(ns-1)+i].dat.data_ro
