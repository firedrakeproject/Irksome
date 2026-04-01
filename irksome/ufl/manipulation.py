"""Manipulation of expressions containing :class:`~.TimeDerivative`
terms.

These can be used to do some basic checking of the suitability of a
:class:`~ufl.Form` for use in Irksome (via :func:`~.check_integrals`), and
splitting out terms in the :class:`~ufl.Form` that contain a time
derivative from those that don't (via :func:`~.extract_terms`).
"""
from functools import singledispatchmethod
from itertools import chain
from typing import NamedTuple, List, Sequence

from ufl.algorithms import extract_coefficients
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.classes import (
    BaseForm, Coefficient,
    Expr, Form, FormSum, Integral,
    Division, Product, Dot, Inner, Outer,
    PositiveRestricted, NegativeRestricted,
    CellAvg, FacetAvg, Conj, Derivative,
    Variable, Sum, ListTensor,
)

from .deriv import TimeDerivative

__all__ = ("SplitTimeForm", "check_integrals", "extract_terms")


class SplitTimeForm(NamedTuple):
    """A container for a form split into time terms and a remainder."""
    time: Form
    remainder: Form


class TimeDerivativeChecker(DAGTraverser):
    """Check that TimeDerivative appears linearly and return the Coefficients
       under TimeDerivatives.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(Integral)
    def integral(self, o):
        return self(o.integrand())

    @process.register(TimeDerivative)
    def time_derivative(self, o):
        f, = o.ufl_operands
        return tuple(extract_coefficients(f))

    @process.register(Expr)
    @DAGTraverser.postorder
    def nonlinear_op(self, o, *ops):
        if any(ops):
            raise ValueError("Can't apply nonlinear operator to time derivative")
        return ()

    @process.register(Division)
    @process.register(Product)
    @process.register(Inner)
    @process.register(Dot)
    @process.register(Outer)
    @DAGTraverser.postorder
    def product(self, o, a, b):
        if a and b:
            raise ValueError("Can't take product of time derivatives")
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
    @DAGTraverser.postorder
    def linear_op(self, o, *ops):
        return tuple(set(chain(*ops)))


def check_integrals(integrals: List[Integral],
                    timedep_coeffs: Sequence[Coefficient] = (),
                    expect_time_derivative: bool = True) -> List[Integral]:
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

    mapper = TimeDerivativeChecker()
    time_derivatives = set(chain.from_iterable(map(mapper, integrals)))

    if expect_time_derivative and time_derivatives != set(timedep_coeffs):
        raise ValueError(f"Expecting 1 TimeDerivative, not {len(time_derivatives)}")

    if not expect_time_derivative and len(time_derivatives & set(timedep_coeffs)) > 0:
        raise ValueError("Not expecting a TimeDerivative of this coefficient")

    return integrals


class TimeDerivativeCoefficientFinder(DAGTraverser):
    """Determines whether an expression depends on TimeDerivative of a coefficient
    """
    def __init__(self, timedep_coeffs, **kwargs):
        super().__init__(**kwargs)
        self.timedep_coeffs = set(timedep_coeffs)

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(BaseForm)
    @process.register(Expr)
    @DAGTraverser.postorder
    def generic(self, o, *ops):
        return any(ops)

    @process.register(Integral)
    def integral(self, o):
        return self(o.integrand())

    @process.register(TimeDerivative)
    def time_derivative(self, o):
        f, = o.ufl_operands
        terminals = set(traverse_unique_terminals(f))
        return len(terminals & self.timedep_coeffs) > 0


def extract_terms(form: Form, timedep_coeffs: Sequence[Coefficient] = ()) -> SplitTimeForm:
    """Extract terms from a :class:`~ufl.Form`.

    This splits a form (a sum of integrals) into those integrals which
    do contain a :class:`~.TimeDerivative` acting on `u0` and those that don't.

    :arg form: The form to split.
    :arg timedep_coeffs: The time-dependent coefficients.
    :returns: a :class:`~.SplitTimeForm` tuple.
    :raises ValueError: if the form does not apply anything other than
        first-order time derivatives to a single coefficient.
    """
    remainder = Form([])
    if isinstance(form, FormSum):
        # Assume that TimeDerivative cannot occur on BaseForms
        terms = form.components()
        remainder = sum(f for f in terms if not isinstance(f, Form))
        form = sum(f for f in terms if isinstance(f, Form))

    time_finder = TimeDerivativeCoefficientFinder(timedep_coeffs)
    time_terms = []
    rest_terms = []
    for itg in form.integrals():
        if time_finder(itg):
            time_terms.append(itg)
        else:
            rest_terms.append(itg)

    time_terms = check_integrals(time_terms, timedep_coeffs, expect_time_derivative=True)
    rest_terms = check_integrals(rest_terms, timedep_coeffs, expect_time_derivative=False)
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


def strip_dt_form(F):
    """Helper function to strip all time derivatives from a Form"""
    stripper = TimeDerivativeRemover()

    # Strip dt from all the integrals in the form
    Fnew = stripper(F)

    # Return the form stripped of its time derivatives
    return Fnew
