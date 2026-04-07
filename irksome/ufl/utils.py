"""Utility functions for UFL expression analysis in Irksome."""

import numpy
from ufl.algorithms.analysis import extract_type
from ufl.coefficient import Coefficient
from ufl.corealg.traversal import traverse_unique_terminals


def has_coefficient_in_expr(expr, coeff):
    """Check if a UFL Coefficient appears as a terminal in an expression.

    :arg expr: A UFL expression to search within.
    :arg coeff: A :class:`~ufl.coefficient.Coefficient` to look for.
    :returns: True if ``coeff`` appears as a terminal in ``expr``.
    """
    for term in traverse_unique_terminals(expr):
        if isinstance(term, Coefficient) and term == coeff:
            return True
    return False


def has_composite_time_derivative(form, u):
    """Check if a form contains Dt(g(u)) where g is not the identity.

    Returns True if any TimeDerivative operand is a composite expression
    involving ``u`` rather than ``u`` itself (or a component of ``u``).

    :arg form: A :class:`~ufl.Form` to inspect.
    :arg u: The prognostic :class:`~firedrake.Function`.
    :returns: True if the form contains a composite time derivative.
    """
    from .deriv import TimeDerivative  # local to avoid circular import

    derivs = extract_type(form, TimeDerivative)
    ubits = set(u[i] for i in numpy.ndindex(u.ufl_shape))
    for td in derivs:
        op, = td.ufl_operands
        op_bits = set(op[i] for i in numpy.ndindex(op.ufl_shape))
        if not op_bits.issubset(ubits) and has_coefficient_in_expr(op, u):
            return True
    return False
