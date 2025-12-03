from functools import singledispatchmethod

from ufl.corealg.dag_traverser import DAGTraverser

from ufl.constantvalue import IntValue
from ufl.classes import (
    Argument, ConstantValue, Coefficient, SpatialCoordinate,
    Abs, Conj, Curl, Derivative, Div, Grad, Indexed, ReferenceGrad,
    ReferenceValue, Variable, ComponentTensor, IndexSum, Skew, Sym, Trace,
    Transposed, Determinant, Inverse, Division, Product, Inner, Dot, Outer,
    Cross, Sum, ListTensor, ExprList, ExprMapping, Power, MathFunction,
    Conditional, Condition, MultiIndex, MaxValue, MinValue, Form, Integral,
)

from .deriv import TimeDerivative


class TimeDegreeEstimator(DAGTraverser):
    """Time degree estimator.

    This algorithm is exact for a few operators and heuristic for many.
    """
    def __init__(self, test_degree, trial_degree, t=None, timedep_coeffs=None, **kwargs):
        super().__init__(**kwargs)
        self.test_degree = test_degree
        self.trial_degree = trial_degree
        self.t = t
        self.timedep_coeffs = timedep_coeffs

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(Argument)
    def argument(self, o):
        return self.test_degree if o.number() == 0 else self.trial_degree

    @process.register(ConstantValue)
    def constant(self, o):
        if self.t is not None and o is self.t:
            return 1
        else:
            return 0

    @process.register(Coefficient)
    @process.register(SpatialCoordinate)
    def terminal(self, o):
        if self.t is not None and o is self.t:
            return 1
        elif self.timedep_coeffs is None or o in self.timedep_coeffs:
            return self.trial_degree
        else:
            return 0

    @process.register(TimeDerivative)
    @DAGTraverser.postorder
    def time_derivative(self, o, degree):
        return max(degree - 1, 0)

    @process.register(Abs)
    @process.register(Conj)
    @process.register(Curl)
    @process.register(Derivative)
    @process.register(Div)
    @process.register(Grad)
    @process.register(Indexed)
    @process.register(ReferenceGrad)
    @process.register(ReferenceValue)
    @process.register(Variable)
    @process.register(ComponentTensor)
    @process.register(IndexSum)
    @process.register(Skew)
    @process.register(Sym)
    @process.register(Trace)
    @process.register(Transposed)
    @DAGTraverser.postorder
    def terminal_modifier(self, o, degree, *ops):
        return degree

    @process.register(Determinant)
    @process.register(Inverse)
    @DAGTraverser.postorder
    def not_handled(self, v, *ops):
        raise NotImplementedError(f"Degree estimation for {type(v).__name__} not handled.")

    @process.register(Division)
    @process.register(Product)
    @process.register(Inner)
    @process.register(Dot)
    @process.register(Outer)
    @process.register(Cross)
    @DAGTraverser.postorder
    def add_degrees(self, v, *ops):
        return sum(ops)

    @process.register(Sum)
    @process.register(ListTensor)
    @process.register(ExprList)
    @process.register(ExprMapping)
    @DAGTraverser.postorder
    def max_degree(self, v, *ops):
        return max(ops)

    @process.register(Power)
    @DAGTraverser.postorder
    def power(self, v, a, b):
        """Apply to power.

        If b is a positive integer: degree(a**b) == degree(a)*b
        otherwise use the heuristic: degree(a**b) == degree(a) + 2.
        """
        _f, g = v.ufl_operands
        if isinstance(g, IntValue):
            gi = g.value()
            if gi >= 0:
                return a * gi
        # Something to a non-(positive integer) power, e.g. float,
        # negative integer, Coefficient, etc.
        return a + 2

    @process.register(MathFunction)
    @DAGTraverser.postorder
    def math_function(self, v, a):
        """Apply to math_function.

        Using the heuristic:
        degree(sin(const)) == 0
        degree(sin(a)) == degree(a)+2
        which can be wildly inaccurate but at least gives a somewhat
        high integration degree.
        """
        if a:
            return a + 2
        else:
            return a

    @process.register(Conditional)
    @DAGTraverser.postorder
    def conditional(self, v, c, *ops):
        return max(*ops)

    @process.register(MinValue)
    @process.register(MaxValue)
    @DAGTraverser.postorder
    def minmax(self, v, *ops):
        return max(*ops)

    @process.register(Condition)
    @process.register(MultiIndex)
    @DAGTraverser.postorder
    def non_numeric(self, v, *args):
        return None


def estimate_time_degree(expression, test_degree, trial_degree, t=None, timedep_coeffs=None):
    de = TimeDegreeEstimator(test_degree, trial_degree, t=t, timedep_coeffs=timedep_coeffs)
    if isinstance(expression, Form):
        if not expression.integrals():
            return 0
        degree = max(map(de, (it.integrand() for it in expression.integrals())))
    elif isinstance(expression, Integral):
        degree = de(expression.integrand())
    else:
        degree = de(expression)
    default_degree = test_degree + trial_degree
    return default_degree if degree is None else degree
