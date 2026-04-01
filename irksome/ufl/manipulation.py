"""Manipulation of expressions containing :class:`~.TimeDerivative`
terms.

These can be used to do some basic checking of the suitability of a
:class:`~ufl.Form` for use in Irksome (via :func:`~.check_integrals`), and
splitting out terms in the :class:`~ufl.Form` that contain a time
derivative from those that don't (via :func:`~.extract_terms`).
"""
from functools import singledispatchmethod
from typing import NamedTuple, List

from ufl.corealg.dag_traverser import DAGTraverser
from ufl.classes import (Argument, BaseForm,
                         Cofunction, Coefficient, ConstantValue,
                         Expr, Form, FormSum, Integral, SpatialCoordinate)

from .deriv import TimeDerivative

__all__ = ("SplitTimeForm", "check_integrals", "extract_terms")


class SplitTimeForm(NamedTuple):
    """A container for a form split into time terms and a remainder."""
    time: Form
    remainder: Form


class TimeDerivativeChecker(DAGTraverser):
    """Determines whether an expression depends on a set of coefficients.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(BaseForm)
    @process.register(Expr)
    @DAGTraverser.postorder
    def generic(self, o, *ops):
        return sum(ops)

    @process.register(TimeDerivative)
    @DAGTraverser.postorder
    def time_derivative(self, o, op):
        return op + 1

    @process.register(Integral)
    def integral(self, o):
        return self(o.integrand())

    @process.register(Form)
    def form(self, o):
        return sum(self(itg) for itg in o.integrals())


def check_integrals(integrals: List[Integral], expect_time_derivative: bool = True) -> List[Integral]:
    """Check a list of integrals for linearity in the time derivative.

    :arg integrals: list of integrals.
    :arg expect_time_derivative: Are we expecting to see a time
        derivative?
    :raises ValueError: if we are expecting a time derivative and
        don't see one, or time derivatives are applied nonlinearly, to
        more than one coefficient, or more than first order."""
    return integrals
    # TODO
    mapper = TimeDerivativeChecker()
    time_derivatives = 0
    for integral in integrals:
        time_derivatives += mapper(integral)
    howmany = int(expect_time_derivative)
    if len(time_derivatives - {()}) != howmany:
        raise ValueError(f"Expecting time derivative applied to {howmany}"
                         f"coefficients, not {len(time_derivatives - {()})}")
    return integrals


class CoefficientFinder(DAGTraverser):
    """Determines whether an expression depends on a set of coefficients.
    """
    def __init__(self, timedep_coeffs=None, **kwargs):
        super().__init__(**kwargs)
        if timedep_coeffs is None:
            timedep_coeffs = {}
        self.timedep_coeffs = timedep_coeffs

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

    @process.register(Argument)
    @process.register(Cofunction)
    @process.register(Coefficient)
    @process.register(ConstantValue)
    @process.register(SpatialCoordinate)
    def terminal(self, o):
        return o in self.timedep_coeffs


class TimeDerivativeCoefficientFinder(DAGTraverser):
    """Determines whether an expression depends on TimeDerivative of a coefficient
    """
    def __init__(self, timedep_coeffs, **kwargs):
        super().__init__(**kwargs)
        self.rules = CoefficientFinder(timedep_coeffs=timedep_coeffs)

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
        return self.rules(f)


def extract_terms(form: Form, timedep_coeffs: List[Coefficient]) -> SplitTimeForm:
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

    time_finder = TimeDerivativeCoefficientFinder(timedep_coeffs=timedep_coeffs)
    time_terms = []
    rest_terms = []
    for itg in form.integrals():
        if time_finder(itg):
            time_terms.append(itg)
        else:
            rest_terms.append(itg)

    time_terms = check_integrals(time_terms, expect_time_derivative=True)
    rest_terms = check_integrals(rest_terms, expect_time_derivative=False)
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
    """Helper function to strip all time derivatives from a form"""
    stripper = TimeDerivativeRemover()

    # Strip dt from all the integrals in the form
    Fnew = stripper(F)

    # Return the form stripped of its time derivatives
    return Fnew
