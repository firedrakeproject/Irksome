from firedrake import *
from irksome.deriv import TimeDerivative, Dt
from ufl.core.expr import Expr
from ufl.algebra import Product, Sum
from ufl.tensoralgebra import Inner, Dot, Outer
from functools import singledispatch
from gem.node import Memoizer, MemoizerArg
from tsfc.ufl_utils import ufl_reuse_if_untouched

# Single dispatch function to split an expression containing TimeDerivatives of a specific
# function into separate orders of time derivatives on this function
@singledispatch
def split_time_orders_on(e, self, v):
    raise AssertionError(f"Cannot split time orders of {v} on unhandled type {type(e)}")


# Case for time derivatives
@split_time_orders_on.register(TimeDerivative)
def split_on_td(e, self, v):
    o, = e.ufl_operands
    sub_split = self(o, v)
    if len(sub_split) > 1:
        splitted = []
        splitted.append(ufl_reuse_if_untouched(e, sub_split[0]) if sub_split[0] else sub_split[0])
        splitted.append(zero())
        splitted += [ufl_reuse_if_untouched(e, sub_e) if sub_e else sub_e for sub_e in sub_split[1:]]
        return splitted
    else:
        return [ufl_reuse_if_untouched(e, sub_e) if sub_e else sub_e for sub_e in sub_split]


# Case for UFL product-like types
@split_time_orders_on.register(Product)
@split_time_orders_on.register(Inner)
@split_time_orders_on.register(Dot)
@split_time_orders_on.register(Outer)
def split_on_product_like(e, self, v):
    a, b = e.ufl_operands
    splitted_a = self(a, v)
    splitted_b = self(b, v)

    # Compute the length of the splitted form
    splitted_len = max(len(splitted_a), len(splitted_b))
    if not (len(splitted_a) == 1 or len(splitted_b) == 1):
        print("Warning: nonlinear form in the passed expression")

    splitted_product = [zero() for _ in range(splitted_len)]

    # Compose new orders from distributivity of sub-operands
    for i, x in enumerate(splitted_a):
        for j, y in enumerate(splitted_b):
            # Ignore the zero expressions
            if x and y:
                splitted_product[max(i, j)] += ufl_reuse_if_untouched(e, x, y)
    
    return splitted_product


# Case for UFL sum types
@split_time_orders_on.register(Sum)
def split_on_sum(e, self, v):
    a, b = e.ufl_operands
    splitted_a = self(a, v)
    splitted_b = self(b, v)
    order = max(len(splitted_a), len(splitted_b))
    splitted_sum = [zero() for _ in range(order)]

    # Compose new orders from sum of orders
    for i in range(order):
        if i < len(splitted_a):
            splitted_sum[i] = splitted_sum[i] + splitted_a[i]
        if i < len(splitted_b):
            splitted_sum[i] = splitted_sum[i] + splitted_b[i]
    
    return splitted_sum


# Case for general expressions
@split_time_orders_on.register(Expr)
def split_on_expr(e, self, v):
    # Base case if you arrive at the desired expression
    if e == v:
        return [zero(), v]

    os = e.ufl_operands
    if not os:
        # Reached the end of an expression tree branch
        return [e]
    
    # For the unsupported expression, shift to the maximum time order
    splits = [self(o, v) for o in os]
    max_order = 0
    for split in splits:
        max_order = max(max_order, len(split) - 1)
    
    # If no Dt contained, -1th order in Dt(v)
    return [zero() for _ in range(max_order)] + [ufl_reuse_if_untouched(e, *os)]


# Utility function that uses the dispatch to split 
def split_time_form_on(F, v):
    if not F:
        # The form is zero so there is nothing to split
        return [F]
    
    splitted_form = []
    splitter = MemoizerArg(split_time_orders_on)
    
    # Iterate over each integral in the form expression
    for i in F.integrals():
        splitted_expr = splitter(i.integrand(), v)
        if len(splitted_expr) > len(splitted_form):
            splitted_form += [zero() for _ in range(len(splitted_expr) - len(splitted_form))]
        
        # Integrate each split component with the original measure
        for k, e in enumerate(splitted_expr):
            if e:
                splitted_form[k] += Form([i.reconstruct(integrand=e)])
    
    # Each integral in the form is now splitted
    return splitted_form


# Single dispatch function to split an expression containing TimeDerivatives into separate
# orders of this time derivative without conditions on the function

# Base case for unsupported types
@singledispatch
def split_time_orders(e, self):
    raise AssertionError(f"Cannot split time orders on unhandled type {type(e)}")


# Case for time derivatives, increasing the dt-order of all orders of its operand
@split_time_orders.register(TimeDerivative)
def split_td(e, self):
    o, = e.ufl_operands
    sub_split = self(o)
    splitted = [zero()]

    for sub_e in sub_split:
        if sub_e:
            # Only apply the time derivative if differentiating non-zero
            splitted.append(ufl_reuse_if_untouched(e, sub_e))
        else:
            splitted.append(sub_e)
    
    return splitted


# Case for product like UFL expression types
@split_time_orders.register(Product)
@split_time_orders.register(Inner)
@split_time_orders.register(Dot)
@split_time_orders.register(Outer)
def split_product_like(e, self):
    a, b = e.ufl_operands
    splitted_a = self(a)
    splitted_b = self(b)
    splitted_product = [zero() for _ in range(len(splitted_a) + len(splitted_b) - 1)]

    # Compose new orders from distributivity of sub-operands
    for i, x in enumerate(splitted_a):
        for j, y in enumerate(splitted_b):
            splitted_product[i + j] += ufl_reuse_if_untouched(e, x, y)
    
    return splitted_product


# Case for sums
@split_time_orders.register(Sum)
def split_sum(e, self):
    a, b = e.ufl_operands
    splitted_a = self(a)
    splitted_b = self(b)
    order = max(len(splitted_a), len(splitted_b))
    splitted_sum = [zero() for _ in range(order)]

    # Compose new orders from sum of orders
    for i in range(order):
        if i < len(splitted_a):
            splitted_sum[i] += splitted_a[i]
        if i < len(splitted_b):
            splitted_sum[i] += splitted_b[i]
    
    return splitted_sum


# Base case for general expression
@split_time_orders.register(Expr)
def split_expr(e, self):
    os = e.ufl_operands

    if not os:
        # Reached the end of an expression tree branch
        return [e]
    
    # Otherwise check that the unsupported expression doesn't contain time derivatives
    splits = [self(o) for o in os]
    for split in splits:
        if len(split) != 1:
            raise AssertionError(f"Cannot split expression with Dt in unsupported type {type(e)}, "
                                  "rewrite expression in supported way.")
    
    # If no Dt contained, 0th order in Dt
    return [ufl_reuse_if_untouched(e, *os)]


# Utility function to split a form into its different time orders
# The invariant is that the sum of each splitted sum order retrieves the original expression
def split_time_form(F):
    if not F:
        return [F]
    
    splitted_form = []
    splitter = Memoizer(split_time_orders)
    
    # Iterate over each integral in the form expression
    for i in F.integrals():
        splitted_expr = splitter(i.integrand())
        if len(splitted_expr) > len(splitted_form):
            for _ in range(len(splitted_expr) - len(splitted_form)):
                splitted_form.append(zero())
        
        # Integrate each split component with the original measure
        for k, e in enumerate(splitted_expr):
            if e:
                splitted_form[k] += Form([i.reconstruct(integrand=e)])
    
    # Each integral in the form is now splitted
    return splitted_form