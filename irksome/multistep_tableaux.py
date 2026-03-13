import FIAT
import numpy as np
import math
from .tools import get_lagrange_permutation


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
        return np.abs(self.b[-1]) < 1e-10

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
            a, b = get_weights_BDF(order)

        super().__init__(a, b)


def get_weights_BDF(order):
    if order < 1 or order > 6:
        raise ValueError("BDF only valid for orders 1 through 6")
    uint = FIAT.ufc_simplex(1)
    L = FIAT.Lagrange(uint, order, variant="equispaced")

    _, perm = get_lagrange_permutation(L)

    vals = L.tabulate(1, (1.,))[1,][perm]
    beta = vals[-1]

    a = vals / beta
    b = np.zeros(order+1)
    b[-1] = order / beta

    return a, b


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
        b[s - j - 1] = ((-1.0) ** j / (math.factorial(j) * math.factorial(s - j - 1))) * (qwts @ integrand)

    return a, b


def get_weights_AM(s):
    '''compute the weights a, b, for the Adams-Moulton method seeking y^{n + s}.
       s = 0 corresponds to backward Euler--handled separately'''

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
        b[s - j] = ((-1.0) ** j / (math.factorial(j) * math.factorial(s - j))) * (qwts @ integrand)

    return a, b
