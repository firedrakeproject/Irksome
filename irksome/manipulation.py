"""Manipulation of expressions containing :class:`~.TimeDerivative`
terms.

These can be used to do some basic checking of the suitability of a
:class:`~ufl.Form` for use in Irksome (via :func:`~.check_integrals`), and
splitting out terms in the :class:`~ufl.Form` that contain a time
derivative from those that don't (via :func:`~.extract_terms`).
"""
from functools import partial, singledispatch
from itertools import chain
from operator import contains, or_
from typing import Callable, FrozenSet, List, NamedTuple, Tuple, Union

from gem.node import Memoizer
from tsfc.ufl_utils import ufl_reuse_if_untouched
from ufl.algebra import Conj, Division, Product, Sum
from ufl.averaging import CellAvg, FacetAvg
from ufl.coefficient import Coefficient
from ufl.constantvalue import Zero
from ufl.core.expr import Expr
from ufl.core.operator import Operator
from ufl.core.terminal import Terminal
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.differentiation import Derivative
from ufl.form import Form
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.integral import Integral
from ufl.restriction import NegativeRestricted, PositiveRestricted
from ufl.tensoralgebra import Dot, Inner, Outer
from ufl.tensors import ComponentTensor, ListTensor
from ufl.variable import Variable

from irksome.deriv import TimeDerivative

__all__ = ("SplitTimeForm", "check_integrals", "extract_terms")


class SplitTimeForm(NamedTuple):
    """A container for a form split into time terms and a remainder."""
    time: Form
    remainder: Form


def _filter(o: Expr, self: Memoizer) -> Expr:
    if not isinstance(o, Expr):
        raise AssertionError(f"Cannot handle term with type {type(o)}")
    if self.predicate(o):
        return Zero(shape=o.ufl_shape,
                    free_indices=o.ufl_free_indices,
                    index_dimensions=o.ufl_index_dimensions)
    else:
        return ufl_reuse_if_untouched(o, *map(self, o.ufl_operands))


def remove_if(expr: Expr, predicate: Callable[[Expr], bool]) -> Expr:
    """Remove terms from an expression that match a predicate.

    This is done by replacing matching terms by an
    appropriately-shaped :class:`~.Zero`, so only works to remove
    terms that are linear in the expression.

    :arg expr: the expression to remove terms from.
    :arg predicate: a function that indicates if an expression should
        be kept or not.
    :returns: A potentially new expression with terms matching the
        predicate removed."""
    mapper = Memoizer(_filter)
    mapper.predicate = predicate
    return mapper(expr)


Result = Union[Tuple[()], Tuple[Coefficient, ...]]


@singledispatch
def _check_time_terms(o, self: Memoizer) -> Result:
    raise AssertionError(f"Unhandled type {type(o)}")


@_check_time_terms.register(TimeDerivative)
def _check_timederiv(o: TimeDerivative, self: Memoizer) -> Result:
    op, = o.ufl_operands
    if self(op):
        # op already has a TimeDerivative applied to it
        raise ValueError("Can only handle first-order systems")
    terminals = tuple(set(traverse_unique_terminals(op)))
    if len(terminals) != 1 or not isinstance(terminals[0], Coefficient):
        raise ValueError("Time derivative must apply to a single coefficient")
    return terminals


@_check_time_terms.register(Expr)
def _check_nonlinearop(o: Union[Terminal, Operator], self: Memoizer) -> Result:
    if any(map(self, o.ufl_operands)):
        raise ValueError("Can't apply nonlinear operator to time derivative")
    return ()


@_check_time_terms.register(Division)
def _check_division(o: Division, self: Memoizer) -> Result:
    a, b = map(self, o.ufl_operands)
    if b:
        raise ValueError("Can't divide by time derivative")
    return a


@_check_time_terms.register(Product)
@_check_time_terms.register(Inner)
@_check_time_terms.register(Dot)
@_check_time_terms.register(Outer)
def _check_product(o: Operator, self: Memoizer) -> Result:
    a, b = map(self, o.ufl_operands)
    if a and b:
        raise ValueError("Can't take product of time derivatives")
    return a or b


@_check_time_terms.register(PositiveRestricted)
@_check_time_terms.register(NegativeRestricted)
@_check_time_terms.register(CellAvg)
@_check_time_terms.register(FacetAvg)
@_check_time_terms.register(Conj)
@_check_time_terms.register(Derivative)
@_check_time_terms.register(Variable)
@_check_time_terms.register(Sum)
@_check_time_terms.register(ListTensor)
def _check_linearop(o: Operator, self: Memoizer) -> Result:
    return tuple(set(chain(*map(self, o.ufl_operands))))


@_check_time_terms.register(Indexed)
@_check_time_terms.register(IndexSum)
@_check_time_terms.register(ComponentTensor)
def _check_indexed(o: Operator, self: Memoizer) -> Result:
    return self(o.ufl_operands[0])


def check_integrals(integrals: List[Integral], expect_time_derivative: bool = True) -> List[Integral]:
    """Check a list of integrals for linearity in the time derivative.

    :arg integrals: list of integrals.
    :arg expect_time_derivative: Are we expecting to see a time
        derivative?
    :raises ValueError: if we are expecting a time derivative and
        don't see one, or time derivatives are applied nonlinearly, to
        more than one coefficient, or more than first order."""
    mapper = Memoizer(_check_time_terms)
    time_derivatives = set()
    for integral in integrals:
        time_derivatives.update(mapper(integral.integrand()))
    howmany = int(expect_time_derivative)
    if len(time_derivatives - {()}) != howmany:
        raise ValueError(f"Expecting time derivative applied to {howmany}"
                         f"coefficients, not {len(time_derivatives - {()})}")
    return integrals


def summands(o: Expr) -> FrozenSet[Expr]:
    """Flatten a sum tree into a set of summands

    :arg o: the expression to flatten.
    :returns: a frozenset of the summands such that sum(r) == o (up to
        order of arguments)."""
    if isinstance(o, Sum):
        return or_(*map(summands, o.ufl_operands))
    else:
        return frozenset([o])


def extract_terms(form: Form) -> SplitTimeForm:
    """Extract terms from a :class:`~ufl.Form`.

    This splits a form (a sum of integrals) into those integrals which
    do contain a :class:`~.TimeDerivative` and those that don't.

    :arg form: The form to split.
    :returns: a :class:`~.SplitTimeForm` tuple.
    :raises ValueError: if the form does not apply anything other than
        first-order time derivatives to a single coefficient.
    """
    time_terms = []
    rest_terms = []
    for integral in form.integrals():
        integrand = integral.integrand()
        rest = remove_if(integrand, lambda o: isinstance(o, TimeDerivative))
        time = remove_if(integrand, partial(contains, summands(rest)))
        if not isinstance(time, Zero):
            time_terms.append(integral.reconstruct(integrand=time))
        if not isinstance(rest, Zero):
            rest_terms.append(integral.reconstruct(integrand=rest))

    time_terms = check_integrals(time_terms, expect_time_derivative=True)
    rest_terms = check_integrals(rest_terms, expect_time_derivative=False)
    return SplitTimeForm(time=Form(time_terms), remainder=Form(rest_terms))


# Helper function to strip the time derivative from expressions, base case
@singledispatch
def strip_dt(e, self):
    os = e.ufl_operands
    if os:
        stripped_os = map(self, os)
        return ufl_reuse_if_untouched(e, *stripped_os)
    return e


# Case for time derivatives, returning the operand
@strip_dt.register(TimeDerivative)
def strip_dt_td(e, self):
    o, = e.ufl_operands
    return self(o)


# Helper function to strip all time derivatives from a form
def strip_dt_form(F):
    if isinstance(F, Zero):
        # Avoid applying the time derivative stripper to zero forms
        return F

    stripper = Memoizer(strip_dt)

    # Strip dt from all the integrals in the form
    Fnew = Form([i.reconstruct(integrand=stripper(i.integrand())) for i in F.integrals()])

    # Return the form stripped of its time derivatives
    return Fnew
