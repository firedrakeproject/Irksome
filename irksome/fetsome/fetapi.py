from firedrake import UnitIntervalMesh, FunctionSpace, DirichletBC

# Small user-side wrapper to avoid confusions with CG / DG which in Fetsome is dealt with
# by the stepper-generator pair instead of the function space
def TimeFunctionSpace(kt):
    interval = UnitIntervalMesh(1)
    if kt == 0:
        return FunctionSpace(interval, "DG", 0)
    return FunctionSpace(interval, "CG", kt)

# User side wrapper for time dependent DirichletBCs that stores the original expression,
# so that different time points can be interpolated
class TimeDirichletBC(DirichletBC):
    def __init__(self, V, g, sub_domain, method=None):
        self.time_g_expr = g
        super().__init__(V, g, sub_domain, method)
    
    # Obtain the original time dependent expression of the BC
    def get_time_expr(self):
        return self.time_g_expr