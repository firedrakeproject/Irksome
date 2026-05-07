
from functools import singledispatchmethod

from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.domain import as_domain
from ufl.algorithms.map_integrands import map_integrands
from ufl.algorithms import expand_derivatives, replace
from ufl import as_tensor, diff, dx, inner, outer
from ufl.classes import Expr, BaseForm
from ufl import Coefficient

import numpy as np


@ufl_type(
    num_ops=1,
    inherit_shape_from_operand=0,
    inherit_indices_from_operand=0,
)
class TimeProjector(Operator):
    __slots__ = ("order", "quadrature")

    def __init__(self, expression, order, Q):
        self.order = order
        self.quadrature = Q
        Operator.__init__(self, operands=(expression,))

    def _ufl_expr_reconstruct_(self, *operands):
        """Return a new object of the same type with new operands."""
        return TimeProjector(*operands, self.order, self.quadrature)


class TimeProjectorDispatcher(DAGTraverser):
    def __init__(self, element, t, dt, u0, u1, stages, phi):
        super().__init__()
        self.L_trial = element
        self.t = t
        self.dt = dt
        self.u0 = u0
        self.u1 = u1
        self.stages = stages
        self.phi = np.reshape(phi, (-1,))

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(Expr)
    @process.register(BaseForm)
    def expr(self, o):
        return self.reuse_if_untouched(o)

    @process.register(TimeProjector)
    def time_projector(self, o):
        from FIAT import Legendre
        from firedrake import TestFunction, TensorFunctionSpace
        from irksome.galerkin_stepper import getTermGalerkin
        from irksome.constant import vecconst
        # use the internal copy of the state, so it does not get updated again in the outer quadrature
        order = o.order
        Q = o.quadrature
        assert order+1 <= len(self.phi)
        f, = o.ufl_operands
        mesh = as_domain(self.u0.function_space().mesh())
        R = TensorFunctionSpace(mesh, "DG", 0, shape=f.ufl_shape)
        F = inner(f, TestFunction(R))*dx
        F = replace(F, {self.u0: self.u1})

        # compute the hierarchical mass matrix (always the identity)
        ref_el = self.L_trial.get_reference_element()
        L_test = Legendre(ref_el, order)
        test_vals = L_test.tabulate(0, Q.get_points())[(0,)]
        M = np.multiply(test_vals, Q.get_weights()) @ test_vals.T
        Minv = vecconst(np.linalg.inv(M))

        # compute modal expansion tested against c
        c = Coefficient(R)

        test = outer(as_tensor(Minv[:order+1] @ self.phi) / self.dt, c)
        Fc = getTermGalerkin(F, self.L_trial, L_test, Q, self.t, self.dt, self.u1, self.stages, test)

        # compute the L2-Riesz representation by undoing the integral against the test coefficient
        fc = sum(it.integrand() for it in Fc.integrals())
        fproj = expand_derivatives(diff(fc, c))
        return fproj


def expand_time_projectors(expression, element, t, dt, u0, u1, stages, phi):
    rules = TimeProjectorDispatcher(element, t, dt, u0, u1, stages, phi)
    return map_integrands(rules, expression)
