from ufl.differentiation import Derivative
from ufl.log import error
from ufl.core.ufl_type import ufl_type
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags, map_expr_dag
from ufl.algorithms.apply_derivatives import GenericDerivativeRuleset


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


def Dt(f):
    """Short-hand function to produce a :class:`TimeDerivative` of the
    input."""
    return TimeDerivative(f)


class TimeDerivativeRuleset(GenericDerivativeRuleset):
    """Apply AD rules to time derivative expressions.  WIP"""
    def __init__(self, t, timedep_coeffs):
        GenericDerivativeRuleset.__init__(self, ())
        self.t = t
        self.timedep_coeffs = timedep_coeffs

    def coefficient(self, o):
        if o in self.timedep_coeffs:
            return TimeDerivative(o)
        else:
            return self.independent_terminal(o)

    # def indexed(self, o, Ap, ii):
    #     print(o, type(o))
    #     print(Ap, type(Ap))
    #     print(ii, type(ii))
    #     1/0


# mapping rules to splat out time derivatives so that replacement should
# work on more complex problems.
class TimeDerivativeRuleDispatcher(MultiFunction):
    def __init__(self, t, timedep_coeffs):
        MultiFunction.__init__(self)
        self.t = t
        self.timedep_coeffs = timedep_coeffs

    def terminal(self, o):
        return o

    def derivative(self, o):
        error("Missing derivative handler for {0}.".format(type(o).__name__))

    expr = MultiFunction.reuse_if_untouched

    def grad(self, o):
        from firedrake import grad
        if isinstance(o, TimeDerivative):
            return TimeDerivative(grad(*o.ufl_operands))
        return o

    def div(self, o):
        return o

    def reference_grad(self, o):
        return o

    def coefficient_derivative(self, o):
        return o

    def coordinate_derivative(self, o):
        return o

    def time_derivative(self, o):
        f, = o.ufl_operands
        rules = TimeDerivativeRuleset(self.t, self.timedep_coeffs)
        return map_expr_dag(rules, f)


def apply_time_derivatives(expression, t, timedep_coeffs=[]):
    rules = TimeDerivativeRuleDispatcher(t, timedep_coeffs)
    return map_integrand_dags(rules, expression)
