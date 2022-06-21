from ufl.algorithms.estimate_degrees import SumDegreeEstimator
from ufl.form import Form
from ufl.integral import Integral
from ufl.corealg.map_dag import map_expr_dags

class TimeDegreeEstimator(SumDegreeEstimator):
    def __init__(self, t, kt, element_replace_map):
        self.t = t
        self.kt = kt
        super().__init__(0, element_replace_map) 

    # Constants: constants are constants unless you have a time variable
    def constant_value(self, v):
        if v == self.t:
            return 1
        else:
            return super().constant_value(v)
    
    def constant(self, v):
        if v == self.t:
            return 1
        else:
            return super().constant(v)
    
    # Spatial Coordinates: constants with respect to time
    def spatial_coordinate(self, v):
        return 0
    
    def cell_coordinate(self, v):
        return 0
    
    # Arguments, Coefficients: we take that they have the default time degree
    def argument(self, v):
        return self.kt
    
    def coefficient(self, v):
        return self.kt
    
    # Spatial Derivatives: eliminate the reduction of polynomial degree for spatial dx
    def _ignore_spatial_dx(self, v, f):
        return f
    
    grad = _ignore_spatial_dx
    reference_grad = _ignore_spatial_dx
    nabla_grad = _ignore_spatial_dx
    div = _ignore_spatial_dx
    reference_div = _ignore_spatial_dx
    nabla_div = _ignore_spatial_dx
    curl = _ignore_spatial_dx
    reference_curl = _ignore_spatial_dx

    # Reduce degree for time derivatives
    def _reduce_time_degree(self, v, f):
        return super()._reduce_degree(v, f)

    time_derivative = _reduce_time_degree

# Apply the multi function to the form that needs estimation
def estimate_time_degree(e, t, kt, element_replace_map={}):
    if not e:
        return 0

    de = TimeDegreeEstimator(t, kt, element_replace_map)
    if isinstance(e, Form):
        if not e.integrals():
            raise AssertionError("Cannot estimate time degree of form with no integrals.")
        degrees = map_expr_dags(de, [i.integrand() for i in e.integrals()])
    elif isinstance(e, Integral):
        degrees = map_expr_dags(de, [e.integrand()])
    else:
        degrees = map_expr_dags(de, [e])
    degree = max(degrees) if degrees else kt
    return degree