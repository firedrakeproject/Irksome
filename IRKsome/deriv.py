from ufl.differentiation import Derivative
from ufl.core.ufl_type import ufl_type

@ufl_type(num_ops=1)
class TimeDeriv(Derivative):
    def __new__(cls, f):
        if not isinstance(f, 

def Dt(f):
    return TimeDeriv(f)

