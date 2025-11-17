import FIAT
import numpy as np
import math


class MultistepTableau(object):
    """Top-level class representing a tableau encoding
       a multistep method. It has members

    :arg a: a 1d array containing the coefficients of the previous steps
    :arg b: a 1d array containing the weights of the right hand side of the method
    """
    def __init__(self, a, b):

        self.a = a
        self.b = b

    @property
    def num_steps(self):
        """Return the number of stages the method has."""
        return len(self.b)

    @property
    def is_explicit(self):
        return self.b[-1] < 1e-10

    @property
    def is_implicit(self):
        return not self.is_explicit
    

class MultistepMethod(MultistepTableau):
    def __init__(self, method, order):
        if method == 'AB':
            a, b = get_weights_AB(order)
        elif method == 'AM':
            a, b = get_weights_AM(order)
        else:
            try:
                a, b = multistep_dict[method, order]
            except KeyError:
                raise NotImplementedError("Not a recognized multistep method")
        super().__init__(a, b)

# BDF Methods
BDF1a = np.array([0.0, 1.0])
BDF1b = np.array([0.0, 1.0])

BDF2a = np.array([0.3333333333333333, -1.3333333333333333, 1.0])
BDF2b = np.array([0.0, 0.0, 0.6666666666666666])

BDF3a = np.array([-0.18181818181818182, 0.8181818181818182, -1.6363636363636365, 1.0])
BDF3b = np.array([0.0, 0.0, 0.0, 0.5454545454545454])

BDF4a = np.array([0.12, -0.64, 1.44, -1.92, 1.0])
BDF4b = np.array([0.0, 0.0, 0.0, 0.0, 0.48])

BDF5a = np.array([-0.08759124087591241, 0.5474452554744526, -1.4598540145985401, 2.18978102189781, -2.18978102189781, 1.0])
BDF5b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.43795620437956206])

BDF6a = np.array([0.06802721088435375, -0.4897959183673469, 1.530612244897959, -2.7210884353741496, 3.061224489795918, -2.4489795918367347, 1.0])
BDF6b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40816326530612246])

multistep_dict = {
    ('BDF', 1): (BDF1a, BDF1b),
    ('BDF', 2): (BDF2a, BDF2b),
    ('BDF', 3): (BDF3a, BDF3b),
    ('BDF', 4): (BDF4a, BDF4b),
    ('BDF', 5): (BDF5a, BDF5b),
    ('BDF', 6): (BDF6a, BDF6b),
}


def get_weights_AB(s):
    '''compute the weights a, b, for the Adams-Bashforth method seeking y^{n + s}'''
    assert s > 0

    a = np.zeros(s + 1)
    b = np.zeros(s + 1)

    a[-2] = -1.0
    a[-1] = 1.0

    L = FIAT.ufc_simplex(1)

    Q = FIAT.make_quadrature(L, s)
    qpts = Q.get_points()
    qwts = Q.get_weights()

    integrand = np.zeros(len(qpts))

    for j in range(s):    
        for ptidx in range(len(qpts)):
            prod = 1.0
            for i in range(s):
                if i == j:
                    pass
                else:
                    prod = prod * (qpts[ptidx][0] + i)
            
            integrand[ptidx] = prod
        b[s - j - 1] = ((-1.0) ** j / (math.factorial(j) * math.factorial(s - j - 1))) *(qwts @ integrand)

    return a, b


def get_weights_AM(s):
    '''compute the weights a, b, for the Adams-Moulton method seeking y^{n + s}. 
       s = 0 correspons to backward Euler--handled separately'''

    assert s >= 0

    if s == 0:
        return np.array([-1.0, 1.0]), np.array([0.0, 1.0])
    
    a = np.zeros(s + 1)
    b = np.zeros(s + 1)

    a[-2] = -1.0
    a[-1] = 1.0

    L = FIAT.ufc_simplex(1)

    Q = FIAT.make_quadrature(L, s + 1)
    qpts = Q.get_points()
    qwts = Q.get_weights()

    integrand = np.zeros(len(qpts))

    for j in range(s + 1):    
        for ptidx in range(len(qpts)):
            prod = 1.0
            for i in range(s + 1):
                if i == j:
                    pass
                else:
                    prod = prod * (qpts[ptidx][0] + i - 1)
            
            integrand[ptidx] = prod
        b[s - j] = ((-1.0) ** j / (math.factorial(j) * math.factorial(s - j))) *(qwts @ integrand)

    return a, b
