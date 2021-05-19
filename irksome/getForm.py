import numpy
from firedrake import (TestFunction, Function, Constant,
                       split, DirichletBC, interpolate, project)
from firedrake.dmhooks import push_parent
from ufl import diff
from ufl.algorithms import expand_derivatives
from ufl.classes import Zero
from .deriv import TimeDerivative  # , apply_time_derivatives
from ufl.constantvalue import as_ufl
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.analysis import has_exact_type
from ufl.classes import CoefficientDerivative
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.log import error


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

    # def coefficient_derivative(self, o):
    #     error("Derivatives should be applied before executing replace.")


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
        from ufl.algorithms.ad import expand_derivatives
        e = expand_derivatives(e)

    return map_integrand_dags(MyReplacer(mapping2), e)


def getForm(F, butch, t, dt, u0, bcs=None, bc_type="DAE"):
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
    :arg bc_type: How to manipulate the strongly-enforced boundary
         conditions to derive the stage boundary conditions.  Should
         be a string, either "DAE", which implements BCs as
         constraints in the style of a differential-algebraic
         equation, or "ODE", which takes the time derivative of the
         boundary data and evaluates this for the stage values

    On output, we return a tuple consisting of four parts:

       - Fnew, the :class:`Form`
       - k, the :class:`firedrake.Function` holding all the stages.
         It lives in a :class:`firedrake.FunctionSpace` corresponding to the
         s-way tensor product of the space on which the semidiscrete
         form lives.
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the stages,
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
    v = F.arguments()[0]
    V = v.function_space()
    assert V == u0.function_space()

    A = numpy.array([[Constant(aa) for aa in arow] for arow in butch.A])
    c = numpy.array([Constant(ci) for ci in butch.c])

    num_stages = len(c)
    num_fields = len(V)

    Vbig = numpy.prod([V for i in range(num_stages)])
    # Silence a warning about transfer managers when we
    # coarsen coefficients in V
    push_parent(V.dm, Vbig.dm)
    vnew = TestFunction(Vbig)
    k = Function(Vbig)
    if len(V) == 1:
        u0bits = [u0]
        vbits = [v]
        if num_stages == 1:
            vbigbits = [vnew]
            kbits = [k]
        else:
            vbigbits = split(vnew)
            kbits = split(k)
    else:
        u0bits = split(u0)
        vbits = split(v)
        vbigbits = split(vnew)
        kbits = split(k)

    kbits_np = numpy.zeros((num_stages, num_fields), dtype="object")

    for i in range(num_stages):
        for j in range(num_fields):
            kbits_np[i, j] = kbits[i*num_fields+j]

    Ak = A @ kbits_np

    Fnew = Zero()

    for i in range(num_stages):
        repl = {t: t + c[i] * dt}
        for j, (ubit, vbit, kbit) in enumerate(zip(u0bits, vbits, kbits)):
            repl[ubit] = ubit + dt * Ak[i, j]
            repl[vbit] = vbigbits[num_fields * i + j]
            repl[TimeDerivative(ubit)] = kbits_np[i, j]
            if (len(ubit.ufl_shape) == 1):
                for kk, kbitbit in enumerate(kbits_np[i, j]):
                    repl[TimeDerivative(ubit[kk])] = kbitbit
                    repl[ubit[kk]] = repl[ubit][kk]
                    repl[vbit[kk]] = repl[vbit][kk]
        Fnew += replace(F, repl)

    bcnew = []
    gblah = []

    if bcs is None:
        bcs = []
    if bc_type == "ODE":
        u0_mult_np = numpy.divide(1.0, butch.c, out=numpy.zeros_like(butch.c), where=butch.c != 0)
        u0_mult = numpy.array([Constant(mi/dt) for mi in u0_mult_np])
        for bc in bcs:
            gorig = bc._original_arg
            gfoo = expand_derivatives(diff(gorig, t))
            if len(V) == 1:
                for i in range(num_stages):
                    gcur = replace(gfoo, {t: t + c[i] * dt}) + u0_mult[i]*gorig
                    try:
                        gdat = interpolate(gcur-u0_mult[i]*u0, V)
                        gmethod = lambda g, u: gdat.interpolate(g-u0_mult[i]*u)
                    except:  # noqa: E722
                        gdat = project(gcur-u0_mult[i]*u0, V)
                        gmethod = lambda g, u: gdat.project(g-u0_mult[i]*u)
                    gblah.append((gdat, gcur, gmethod))
                    bcnew.append(DirichletBC(Vbig[i], gdat, bc.sub_domain))
            else:
                sub = bc.function_space_index()
                for i in range(num_stages):
                    gcur = replace(gfoo, {t: t + c[i] * dt}) + u0_mult[i]*gorig
                    try:
                        gdat = interpolate(gcur-u0_mult[i]*u0.sub(sub), V.sub(sub))
                        gmethod = lambda g, u: gdat.interpolate(g-u0_mult[i]*u.sub(sub))
                    except:  # noqa: E722
                        gdat = project(gcur-u0_mult[i]*u0.sub(sub), V.sub(sub))
                        gmethod = lambda g, u: gdat.project(g-u0_mult[i]*u.sub(sub))
                    gblah.append((gdat, gcur, gmethod))
                    bcnew.append(DirichletBC(Vbig[sub + num_fields * i],
                                             gdat, bc.sub_domain))
    elif bc_type == "DAE":
        if butch.Ainv is None:
            raise NotImplementedError("Cannot have DAE BCs for this Butcher Tableau")
        else:
            Ainv = numpy.array([[Constant(aa/dt) for aa in arow] for arow in butch.Ainv])

        u0_mult_np = butch.Ainv@numpy.ones_like(butch.c)
        u0_mult = numpy.array([Constant(mi/dt) for mi in u0_mult_np])
        for bc in bcs:
            gorig = as_ufl(bc._original_arg)
            if len(V) == 1:
                for i in range(num_stages):
                    gcur = 0
                    for j in range(num_stages):
                        gcur += Ainv[i, j] * replace(gorig, {t: t + c[j]*dt})
                    try:
                        gdat = interpolate(gcur-u0_mult[i]*u0, V)
                        gmethod = lambda g, u: gdat.interpolate(g-u0_mult[i]*u)
                    except:  # noqa: E722
                        gdat = project(gcur-u0_mult[i]*u0, V)
                        gmethod = lambda g, u: gdat.project(g-u0_mult[i]*u)
                    gblah.append((gdat, gcur, gmethod))
                    bcnew.append(DirichletBC(Vbig[i], gdat, bc.sub_domain))
            else:
                sub = bc.function_space_index()
                for i in range(num_stages):
                    gcur = 0
                    for j in range(num_stages):
                        gcur += Ainv[i, j] * replace(gorig, {t: t + c[j]*dt})
                    try:
                        gdat = interpolate(gcur-u0_mult[i]*u0.sub(sub), V.sub(sub))
                        gmethod = lambda g, u: gdat.interpolate(g-u0_mult[i]*u.sub(sub))
                    except:  # noqa: E722
                        gdat = project(gcur-u0_mult[i]*u0.sub(sub), V.sub(sub))
                        gmethod = lambda g, u: gdat.project(g-u0_mult[i]*u.sub(sub))
                    gblah.append((gdat, gcur, gmethod))
                    bcnew.append(DirichletBC(Vbig[sub + num_fields * i],
                                             gdat, bc.sub_domain))
    else:
        raise ValueError("Unrecognised bc_type: %s", bc_type)

    return Fnew, k, bcnew, gblah
