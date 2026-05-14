"""Manipulation of expressions containing :class:`~.TimeDerivative`
terms.

These can be used to do some basic checking of the suitability of a
:class:`~ufl.Form` for use in Irksome (via :func:`~.check_integrals`), and
splitting out terms in the :class:`~ufl.Form` that contain a time
derivative from those that don't (via :func:`~.split_time_derivative_terms`).
"""
from functools import singledispatchmethod
from itertools import chain
from operator import or_
from typing import NamedTuple, Sequence, FrozenSet

from ufl import TrialFunction, derivative
from ufl.algorithms import expand_derivatives
from ufl.algorithms.analysis import extract_coefficients, extract_type
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.classes import (
    BaseForm, CellAvg, Coefficient, ComponentTensor,
    Conj, Cross, Derivative, Div, Division, Dot, Expr, FacetAvg,
    Form, FormSum, Grad, Indexed, IndexSum, Inner, Integral,
    ListTensor, MultiIndex, NegativeRestricted, Outer, PositiveRestricted,
    Product, Sum, Variable,
)

from .deriv import TimeDerivative

__all__ = ("SplitTimeForm", "check_integrals", "split_time_derivative_terms",
           "remove_time_derivatives", "has_nonlinear_time_derivative")


def has_nonlinear_time_derivative(F, u0):
    """True iff ``F`` contains a TimeDerivative of an expression that is
    nonlinear in u0 -- i.e. ``Dt(g(u0))`` for some nonlinear g.  These
    cases lose mass conservation when chain-ruled through the
    stage-derivative form, and require the conservative two-evaluation
    discretisation.

    For each ``Dt(f)`` in the form, the Gateaux derivative of ``f`` with
    respect to ``u0`` is taken in a trial direction.  If the derivative
    still depends on ``u0``, ``f`` is nonlinear in u0.  This delegates
    the classification of linear operators (Grad, Div, Indexed,
    restrictions, ListTensor, ComponentTensor, ...) to UFL's own
    derivative machinery rather than maintaining a parallel exemption
    list inside Irksome.

    .. warning::

       The detection is syntactic: it checks whether ``u0`` appears
       under ``Dt`` after differentiation.  If a user creates an
       intermediate :class:`~firedrake.Function` whose values were
       interpolated from an expression in u0 and then writes
       ``Dt(that_intermediate)``, the syntactic dependence on u0 is
       lost and this function will declare the form safe.  The
       resulting discretisation is *not* mass-conservative.  Always
       wrap the symbolic expression directly in ``Dt`` (as
       ``Dt(theta(u))``, not ``Dt(theta_function)``).
    """
    Trial = TrialFunction(u0.function_space())
    for td in extract_type(F, TimeDerivative):
        f, = td.ufl_operands
        if u0 not in extract_coefficients(f):
            # Dt(f(t,x)) -- no u0 dependence, chain-ruled analytically
            continue
        D = expand_derivatives(derivative(f, u0, Trial))
        if u0 in extract_coefficients(D):
            return True
    return False


class SplitTimeForm(NamedTuple):
    """A container for a form split into time terms and a remainder."""
    time: BaseForm
    remainder: BaseForm


class TimeDerivativeChecker(DAGTraverser):
    """Check that TimeDerivative appears linearly and return the Coefficients
       under TimeDerivatives.
    """
    def __init__(self, t, timedep_coeffs, **kwargs):
        super().__init__(**kwargs)
        terminals = []
        for c in timedep_coeffs:
            terminals.extend(ci for ci in traverse_unique_terminals(c) if not isinstance(ci, MultiIndex))
        self.timedep_coeffs = frozenset(terminals)
        self.t = t

    def check_time_dependence(self, expr):
        expr_terminals = frozenset(traverse_unique_terminals(expr))
        return (self.t in expr_terminals) or len(self.timedep_coeffs & expr_terminals) > 0

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(Integral)
    def integral(self, o):
        return self(o.integrand())

    @process.register(TimeDerivative)
    @DAGTraverser.postorder
    def time_derivative(self, o, *ops):
        if any(ops):
            raise ValueError("Can only handle first-order systems")
        f, = o.ufl_operands
        terminals = set(traverse_unique_terminals(f))
        return frozenset(terminals & self.timedep_coeffs)

    @process.register(Expr)
    @DAGTraverser.postorder
    def nonlinear_op(self, o, *ops):
        if any(ops):
            raise ValueError("Can't apply nonlinear operator to TimeDerivative")
        return frozenset()

    @process.register(Division)
    @DAGTraverser.postorder
    def division(self, o, a, b):
        oa, ob = o.ufl_operands
        if b:
            raise ValueError("Can't divide by TimeDerivative")
        if a and self.check_time_dependence(ob):
            raise ValueError("Can't divide TimeDerivative by time-dependent expression")
        return a

    @process.register(Product)
    @process.register(Inner)
    @process.register(Cross)
    @process.register(Dot)
    @process.register(Outer)
    @DAGTraverser.postorder
    def product(self, o, a, b):
        oa, ob = o.ufl_operands
        if a and b:
            raise ValueError("Can't take product of TimeDerivatives")
        if (a and self.check_time_dependence(ob)) or (b and self.check_time_dependence(oa)):
            raise ValueError("Can't take product of TimeDerivative and time-dependent expression")
        return a or b

    @process.register(PositiveRestricted)
    @process.register(NegativeRestricted)
    @process.register(CellAvg)
    @process.register(FacetAvg)
    @process.register(Conj)
    @process.register(Derivative)
    @process.register(Variable)
    @process.register(Sum)
    @process.register(ListTensor)
    @process.register(Indexed)
    @process.register(IndexSum)
    @process.register(ComponentTensor)
    @DAGTraverser.postorder
    def linear_op(self, o, *ops):
        return frozenset(chain(*ops))


def check_integrals(integrals: Sequence[Integral],
                    t: Expr = None,
                    timedep_coeffs: Sequence[Coefficient] = (),
                    expect_time_derivative: bool = True):
    """Check a list of integrals for linearity in the time derivative.

    :arg integrals: list of integrals.
    :arg timedep_coeffs: The time-dependent coefficients.
    :arg expect_time_derivative: Are we expecting to see a time
        derivative?
    :raises ValueError: if we are expecting a time derivative and
        don't see one, or time derivatives are applied nonlinearly, to
        more than one coefficient, or more than first order."""
    if len(integrals) == 0:
        return integrals

    mapper = TimeDerivativeChecker(t, timedep_coeffs)
    time_derivatives = set(chain.from_iterable(map(mapper, integrals)))
    howmany = int(expect_time_derivative)
    if len(time_derivatives) != howmany:
        raise ValueError(f"Expecting time derivative applied to {howmany} "
                         f"coefficients, not {len(time_derivatives)}")


def summands(o: Expr) -> FrozenSet[Expr]:
    """Flatten a sum tree into a set of summands

    :arg o: the expression to flatten.
    :returns: a frozenset of the summands such that sum(r) == o (up to
        order of arguments)."""
    if isinstance(o, Sum):
        return or_(*map(summands, o.ufl_operands))
    else:
        return frozenset([o])


def split_time_derivative_terms(form: BaseForm,
                                t: Expr = None,
                                timedep_coeffs: Sequence[Coefficient] = ()
                                ) -> SplitTimeForm:
    """Split terms from a :class:`~ufl.Form`.

    This splits a form (a sum of integrals) into those integrals which
    do contain a :class:`~.TimeDerivative` acting on `timedep_coeffs` and those that don't.

    :arg form: The form to split.
    :arg t: The time variable.
    :arg timedep_coeffs: The time-dependent coefficients.
    :returns: a :class:`~.SplitTimeForm` tuple.
    :raises ValueError: if the form does not apply anything other than
        first-order time derivatives to a single coefficient.
    """
    remainder = Form([])
    if isinstance(form, FormSum):
        # Assume that TimeDerivative cannot occur on BaseForms
        weights = form.weights()
        components = form.components()
        remainder = sum(w*f for w, f in zip(weights, components) if not isinstance(f, Form))
        form = sum(w*f for w, f in zip(weights, components) if isinstance(f, Form))

    mapper = TimeDerivativeChecker(t, timedep_coeffs)
    rest_terms = []
    time_terms = []
    for integral in form.integrals():
        rest = []
        time = []
        for term in summands(integral.integrand()):
            tcoeffs = mapper(term)
            if len(tcoeffs) == 0:
                rest.append(term)
            else:
                time.append(term)
        if len(rest):
            rest_terms.append(integral.reconstruct(integrand=sum(rest)))
        if len(time):
            time_terms.append(integral.reconstruct(integrand=sum(time)))

    return SplitTimeForm(time=Form(time_terms), remainder=Form(rest_terms)+remainder)


class TimeDerivativeRemover(DAGTraverser):
    """Removes TimeDerivative from an expression"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(Expr)
    def generic(self, o):
        return self.reuse_if_untouched(o)

    @process.register(Integral)
    def integral(self, o):
        return o.reconstruct(integrand=self(o.integrand()))

    @process.register(Form)
    def form(self, o):
        return Form([self(itg) for itg in o.integrals()])

    @process.register(TimeDerivative)
    def time_derivative(self, o):
        f, = o.ufl_operands
        return f


def remove_time_derivatives(F: Form):
    """Helper function to strip all time derivatives from a Form"""
    stripper = TimeDerivativeRemover()

    # Strip dt from all the integrals in the form
    Fnew = stripper(F)

    # Return the form stripped of its time derivatives
    return Fnew
