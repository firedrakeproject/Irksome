from firedrake import *
from irksome.deriv import TimeDerivative, Dt
import sys
from ufl.form import Form
from functools import singledispatch, reduce
from gem.node import Memoizer, MemoizerArg
from tsfc.ufl_utils import ufl_reuse_if_untouched

# Map that translates the name of a generator to its code
translate_generator = {"petrov": "CPG", "tdg": "DG"}

# Available generator families
generator_codes = ["CPG", "DG"]

# Old (and possibly incorrect) generator families
old_generator_codes = ["STCG, NSTCG"]

# Helper function to extract the points corresponding to the
# nodal basis evaluation points of the time finite element
def extract_time_points(T):
    Tfs = T.finat_element.fiat_equivalent
    T_dual_basis = Tfs.dual_basis()

    points = []
    for N in T_dual_basis:
        point_dict = N.get_point_dict()
        point = list(point_dict.keys())[0][0]

        if abs(point) < sys.float_info.epsilon:
            points.append(0.)
        else:
            points.append(point)
    
    return points

# Utility function to compose the expression of a spacetime function evaluated
# at a time point
def spacetime_dot(basis_tau, us):
    if len(basis_tau) != len(us):
        raise AssertionError("Basis size and number of space function coefficients must be the same.")
    u_tau = reduce(lambda x, y: x + y, [Constant(basis_tau[i]) * us[i] for i in range(len(us))])
    return u_tau

# Utility function to rapidly construct Dt(...[n times]...(Dt(u))) of a function u
def dt_n(u, n):
    return Dt(dt_n(u, n - 1)) if n != 0 else u


# Returns the maximum order of a time derivative applied on the given function
@singledispatch
def compute_max_time_order(e, self, v):
    # Base
    if e == v:
        return 0

    os = e.ufl_operands
    if os:
        os_orders = [self(o, v) for o in os]
        return max(os_orders)

    return -1

# Case for time derivatives
@compute_max_time_order.register(TimeDerivative)
def compute_max_time_order_td(e, self, v):
    o, = e.ufl_operands
    o_order = self(o, v)
    return o_order + 1 if o_order > -1 else -1

# Utility function that computes the maximum time derivative order of a given
# expression inside a form
def max_time_order(F, v):
    if not F:
        return -1
    
    order_calculator = MemoizerArg(compute_max_time_order)

    return max([order_calculator(i.integrand(), v) for i in F.integrals()])


# Single dispatch function to remove all TimeDerivatives from an expression

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
    if not F:
        # Avoid applying the time derivative stripper to zero forms
        return F

    stripper = Memoizer(strip_dt)

    # Strip dt from all the integrals in the form
    Fnew = Form([i.reconstruct(integrand=stripper(i.integrand())) for i in F.integrals()])
    
    # Return the form stripped of its time derivatives
    return Fnew