import ufl
from ufl.corealg.map_dag import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags, map_integrands, map_expr_dag
from ufl.classes import Zero, Form, Integral



def map_integrands_filtered(function, form, only_integral_type=None):
    """Apply transform(expression) to each integrand
    expression in form, or to form if it is an Expr.
    Assumes that function returns a (result, bool) pair on integrands, and we
    will keep only the values that are True
    """
    if isinstance(form, Form):
        mapped_integrals = [map_integrands_filtered(function, itg, only_integral_type)
                            for itg in form.integrals()]
        nonzero_integrals = [itg for itg in mapped_integrals
                             if not isinstance(itg.integrand(), Zero)]
        return Form(nonzero_integrals)
    elif isinstance(form, Integral):
        itg = form
        if (only_integral_type is None) or (itg.integral_type() in only_integral_type):
            form_integrand, keepme = function(itg.integrand())
            if keepme:
                return itg.reconstruct(form_integrand)
            else:
                return itg.reconstruct(Zero(shape=form_integrand.ufl_shape,
                                            free_indices=form_integrand.ufl_free_indices,
                                            index_dimensions=form_integrand.ufl_index_dimensions))
        else:
            return itg
    elif isinstance(form, Expr):
        integrand = form
        return function(integrand)
    else:
        error("Expecting Form, Integral or Expr.")

def map_integrand_dags_filtered(function, form, only_integral_type=None, compress=True):
    return map_integrands_filtered(lambda expr: map_expr_dag(function, expr, compress),
                          form, only_integral_type)




class TimeTermsKiller(MultiFunction):
    expr = MultiFunction.reuse_if_untouched
    def time_derivative(self, o):
        return ufl.classes.Zero(o.ufl_shape, o.ufl_free_indices, o.ufl_index_dimensions)


class TimeTermsCollector(MultiFunction):
    def expr(self, o, *ops):
        #print(ops)
        if ops == ():
            return self.reuse_if_untouched(o, *ops), False
        else:
            justops = []
            timep = []
            for x, y in ops:
                justops.append(x)
                timep.append(y)
            return self.reuse_if_untouched(o, *justops), any(timep)

    def terminal(self, o):
        return o, False
        
    def time_derivative(self, o):
        op, = o.ufl_operands
        return self.reuse_if_untouched(o), True

    def sum(self, o, left, right):
        (l, ltime) = left
        (r, rtime) = right
        if ltime == rtime:
            return self.reuse_if_untouched(o, l, r), ltime
        else:
            return l if ltime else r, True

def split_time_terms(F):
    return (map_integrand_dags(TimeTermsKiller(), F),
            map_integrand_dags_filtered(TimeTermsCollector(), F))

