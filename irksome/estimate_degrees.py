from functools import singledispatchmethod

from ufl.corealg.dag_traverser import DAGTraverser

from ufl import as_ufl
from ufl.constantvalue import IntValue
from ufl.classes import (
    Argument, ConstantValue, Coefficient, SpatialCoordinate,
    Abs, Conj, Curl, Derivative, Div, Grad, Indexed, ReferenceGrad,
    ReferenceValue, Variable, ComponentTensor, IndexSum, Skew, Sym, Trace,
    Transposed, Determinant, Inverse, Division, Product, Inner, Dot, Outer,
    Cross, Sum, ListTensor, ExprList, ExprMapping, Power, MathFunction,
    Conditional, Condition, MultiIndex, MaxValue, MinValue, Form, Integral, Label,
    Cofactor, FormSum, Interpolate, Cofunction, BaseForm
)

from .deriv import TimeDerivative


class TimeDegreeEstimator(DAGTraverser):
    """Time degree estimator.

    This algorithm is exact for a few operators and heuristic for many.
    """
    def __init__(self, degree_mapping=None, **kwargs):
        super().__init__(**kwargs)
        if degree_mapping is None:
            degree_mapping = {}
        self.degree_mapping = degree_mapping

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(FormSum)
    def formsum(self, o):
        return max(self(as_ufl(w)) + self(c)
                   for w, c in zip(o.weights(), o.components()))

    @process.register(Interpolate)
    def interpolate(self, o):
        return sum(map(self, o.argument_slots()))

    @process.register(Form)
    def form(self, o):
        return 0 if o.empty() else max(map(self, o.integrals()))

    @process.register(Integral)
    def integral(self, o):
        return self(o.integrand())

    @process.register(Argument)
    @process.register(Cofunction)
    @process.register(Coefficient)
    @process.register(ConstantValue)
    @process.register(SpatialCoordinate)
    def terminal(self, o):
        return self.degree_mapping.get(o, 0)

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
    @DAGTraverser.postorder
    def terminal_modifier(self, o, degree, *ops):
        return degree

    @process.register(Cofactor)
    @process.register(Skew)
    @process.register(Sym)
    @process.register(Trace)
    @process.register(Transposed)
    @process.register(Determinant)
    @process.register(Inverse)
    @DAGTraverser.postorder
    def not_handled(self, v, *ops):
        # We should not be here after preprocessing with apply_algebra_lowering
        raise NotImplementedError(f"Degree estimation for {type(v).__name__} not handled.")

    @process.register(Sum)
    @process.register(ListTensor)
    @process.register(ExprList)
    @process.register(ExprMapping)
    @DAGTraverser.postorder
    def max_degree(self, v, *ops):
        return 0 if len(ops) == 0 else max(ops)

    @process.register(Division)
    @process.register(Product)
    @process.register(Inner)
    @process.register(Dot)
    @process.register(Outer)
    @process.register(Cross)
    @DAGTraverser.postorder
    def add_degrees(self, v, *ops):
        return sum(ops)

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

    @process.register(Label)
    @process.register(Condition)
    @process.register(MultiIndex)
    @DAGTraverser.postorder
    def non_numeric(self, v, *args):
        return None


def get_degree_mapping(expression, test_degree, trial_degree, t=None, timedep_coeffs=None):
    degree_mapping = {}
    if isinstance(expression, BaseForm):
        for arg in expression.arguments():
            degree_mapping[arg] = test_degree if arg.number() == 0 else trial_degree

    if t is not None:
        degree_mapping[t] = 1

    if timedep_coeffs is not None:
        for c in timedep_coeffs:
            degree_mapping[c] = trial_degree
    return degree_mapping


def estimate_time_degree(expression, test_degree, trial_degree, t=None, timedep_coeffs=None):
    degree_mapping = get_degree_mapping(expression, test_degree, trial_degree, t=t, timedep_coeffs=timedep_coeffs)
    degree_estimator = TimeDegreeEstimator(degree_mapping=degree_mapping)
    return degree_estimator(expression)
