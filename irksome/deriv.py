from ufl.constantvalue import as_ufl
from ufl.differentiation import Derivative
from ufl.core.ufl_type import ufl_type
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags, map_expr_dag
from ufl.algorithms.apply_derivatives import GenericDerivativeRuleset
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering


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
        self.timedep_coeffs = timedep_coeffs

    def coefficient(self, o):
        if self.t is not None and o is self.t:
            return as_ufl(1.0)
        elif self.timedep_coeffs is None or o in self.timedep_coeffs:
            return TimeDerivative(o)
        else:
            return self.independent_terminal(o)

    def spatial_coordinate(self, o):
        return self.independent_terminal(o)

    def time_derivative(self, o, f):
        if isinstance(f, TimeDerivative):
            return TimeDerivative(f)
        else:
            return map_expr_dag(self, f)

    def _linear_op(self, o, *operands):
        return o._ufl_expr_reconstruct_(*operands)

    indexed = _linear_op
    derivative = _linear_op
    grad = _linear_op
    curl = _linear_op
    div = _linear_op


# mapping rules to splat out time derivatives so that replacement should
# work on more complex problems.
class TimeDerivativeRuleDispatcher(MultiFunction):
    def __init__(self, t=None, timedep_coeffs=None):
        MultiFunction.__init__(self)
        self.rules = TimeDerivativeRuleset(t=t, timedep_coeffs=timedep_coeffs)

    def time_derivative(self, o):
        f, = o.ufl_operands
        return map_expr_dag(self.rules, f)

    ufl_type = MultiFunction.reuse_if_untouched


def apply_time_derivatives(expression, t=None, timedep_coeffs=None):
    rules = TimeDerivativeRuleDispatcher(t=t, timedep_coeffs=timedep_coeffs)
    return map_integrand_dags(rules, expression)


def expand_time_derivatives(expression, t=None, timedep_coeffs=None):
    expression = apply_algebra_lowering(expression)
    expression = apply_time_derivatives(expression, t=t, timedep_coeffs=timedep_coeffs)
    return expression
