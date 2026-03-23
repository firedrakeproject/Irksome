import FIAT
import numpy as np
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
    def num_total_steps(self):
        """Return the number of steps the method has."""
        return len(self.b)

    @property
    def num_prev_steps(self):
        """Return the number of previous steps the method requires."""
        return len(self.b) - 1

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

    vals = L.tabulate(1, (1.,))[1, ][perm]
    beta = vals[-1]

    a = vals / beta
    b = np.zeros(order+1)
    b[-1] = order / beta

    return a, b


def get_weights_AB(order):

    assert order > 0

    uint = FIAT.ufc_simplex(1)

    nodes = [-i for i in range(order)]
    nodes.reverse()
    nodes = np.reshape(nodes, (-1, 1))

    Q = FIAT.make_quadrature(uint, 2*order)
    qpts = Q.get_points()
    qwts = Q.get_weights()

    L = FIAT.barycentric_interpolation.LagrangePolynomialSet(uint, nodes)

    vals = L.tabulate(qpts, 0)[(0,)]
    b = np.dot(vals, qwts)
    b = np.insert(b, order, 0)

    a = np.zeros(order + 1)
    a[-1] = 1.0
    a[-2] = -1.0

    return a, b


def get_weights_AM(order):

    assert order >= 0

    if order == 0:
        return np.array([-1.0, 1.0]), np.array([0.0, 1.0])

    uint = FIAT.ufc_simplex(1)

    nodes = [1-i for i in range(order+1)]
    nodes.reverse()
    nodes = np.reshape(nodes, (-1, 1))

    Q = FIAT.make_quadrature(uint, 2*order)
    qpts = Q.get_points()
    qwts = Q.get_weights()

    L = FIAT.barycentric_interpolation.LagrangePolynomialSet(uint, nodes)

    vals = L.tabulate(qpts, 0)[(0,)]
    b = np.dot(vals, qwts)

    a = np.zeros(order + 1)
    a[-1] = 1.0
    a[-2] = -1.0

    return a, b
