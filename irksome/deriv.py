from functools import singledispatchmethod

from ufl.constantvalue import as_ufl
from ufl.core.ufl_type import ufl_type
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.algorithms.map_integrands import map_integrands
from ufl.algorithms.apply_derivatives import GenericDerivativeRuleset
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.form import BaseForm
from ufl.classes import (Coefficient, Conj, Curl, ConstantValue, Derivative,
                         Div, Expr, Grad, Indexed, ReferenceGrad,
                         ReferenceValue, SpatialCoordinate, Variable)


@ufl_type(num_ops=1,
          inherit_shape_from_operand=0,
          inherit_indices_from_operand=0)
class TimeDerivative(Derivative):
    """UFL node representing a time derivative of some quantity/field.
    Note: Currently form compilers do not understand how to process
    these nodes.  Instead, Irksome pre-processes forms containing
    `TimeDerivative` nodes."""
    __slots__ = ()

    def __new__(cls, f):
        return Derivative.__new__(cls)

    def __init__(self, f):
        Derivative.__init__(self, (f,))

    def __str__(self):
        return "d{%s}/dt" % (self.ufl_operands[0],)


def Dt(f, order=1):
    """Short-hand function to produce a :class:`TimeDerivative` of a given order."""
    for k in range(order):
        f = TimeDerivative(f)
    return f


class TimeDerivativeRuleset(GenericDerivativeRuleset):
    """Apply AD rules to time derivative expressions."""
    def __init__(self, t=None, timedep_coeffs=None):
        GenericDerivativeRuleset.__init__(self, ())
        self.t = t
        self._Id = as_ufl(1.0)
        self.timedep_coeffs = timedep_coeffs

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(ConstantValue)
    def constant(self, o):
        if self.t is not None and o is self.t:
            return self._Id
        else:
            return self.independent_terminal(o)

    @process.register(Coefficient)
    @process.register(SpatialCoordinate)
    def terminal(self, o):
        if self.t is not None and o is self.t:
            return self._Id
        elif self.timedep_coeffs is None or o in self.timedep_coeffs:
            return TimeDerivative(o)
        else:
            return self.independent_terminal(o)

    @process.register(TimeDerivative)
    @DAGTraverser.postorder
    def time_derivative(self, o, f):
        if isinstance(f, TimeDerivative):
            return TimeDerivative(f)
        else:
            return self(f)

    @process.register(Conj)
    @process.register(Curl)
    @process.register(Derivative)
    @process.register(Div)
    @process.register(Grad)
    @process.register(Indexed)
    @process.register(ReferenceGrad)
    @process.register(ReferenceValue)
    @process.register(Variable)
    @DAGTraverser.postorder
    def terminal_modifier(self, o, *operands):
        return o._ufl_expr_reconstruct_(*operands)


class TimeDerivativeRuleDispatcher(DAGTraverser):
    '''
    Mapping rules to splat out time derivatives so that replacement should
    work on more complex problems.
    '''
    def __init__(self, t=None, timedep_coeffs=None, **kwargs):
        super().__init__(**kwargs)
        self.rules = TimeDerivativeRuleset(t=t, timedep_coeffs=timedep_coeffs)

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(TimeDerivative)
    def time_derivative(self, o):
        f, = o.ufl_operands
        return self.rules(f)

    @process.register(Expr)
    @process.register(BaseForm)
    def _generic(self, o):
        return self.reuse_if_untouched(o)


def apply_time_derivatives(expression, t=None, timedep_coeffs=None):
    rules = TimeDerivativeRuleDispatcher(t=t, timedep_coeffs=timedep_coeffs)
    return map_integrands(rules, expression)


def expand_time_derivatives(expression, t=None, timedep_coeffs=None):
    expression = apply_algebra_lowering(expression)
    expression = apply_time_derivatives(expression, t=t, timedep_coeffs=timedep_coeffs)
    return expression
