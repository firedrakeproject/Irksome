import FIAT
import numpy
from numpy import sqrt


class ButcherTableau(object):
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c

    @property
    def num_stages(self):
        return len(self.b)


class BackwardEuler(ButcherTableau):
    def __init__(self):
        A = numpy.array([[1.0]])
        b = numpy.array([1.0])
        c = numpy.array([1.0])
        super(BackwardEuler, self).__init__(A, b, c)


class CollocationButcherTableau(ButcherTableau):
    def __init__(self, L):
        assert L.ref_el == FIAT.ufc_simplex(1)

        for ell in L.dual.nodes:
            assert isinstance(ell, FIAT.functional.PointEvaluation)

        c = numpy.asarray([list(ell.pt_dict.keys())[0][0]
                           for ell in L.dual.nodes])

        num_stages = len(c)

        Q = FIAT.make_quadrature(L.ref_el, 2*num_stages)
        qpts = Q.get_points()
        qwts = Q.get_weights()

        Lvals = L.tabulate(0, qpts)[0, ]

        # integrates them all!
        b = Lvals @ qwts

        # now for A, which means we have to adjust the interval
        A = numpy.zeros((num_stages, num_stages))
        for i in range(num_stages):
            qpts_i = qpts * c[i]
            qwts_i = qwts * c[i]
            Lvals_i = L.tabulate(0, qpts_i)[0, ]
            A[i, :] = Lvals_i @ qwts_i

        super(CollocationButcherTableau, self).__init__(A, b, c)


class GaussLegendre(CollocationButcherTableau):
    def __init__(self, num_stages):
        assert num_stages > 0
        U = FIAT.ufc_simplex(1)
        L = FIAT.GaussLegendre(U, num_stages - 1)
        super(GaussLegendre, self).__init__(L)


class LobattoIIIA(CollocationButcherTableau):
    def __init__(self, num_stages):
        assert num_stages > 1
        U = FIAT.ufc_simplex(1)
        L = FIAT.GaussLobattoLegendre(U, num_stages - 1)
        super(LobattoIIIA, self).__init__(L)


class Radau23(ButcherTableau):
    def __init__(self):
        A = numpy.array([[5./12, -1./12], [3./4, 1./4]])
        b = numpy.array([3./4, 1./4])
        c = numpy.array([1./3, 1.])
        super(Radau23, self).__init__(A, b, c)
        

class Radau35(ButcherTableau):
    def __init__(self):
        A = numpy.array([[11./45 - 7*sqrt(6)/360, 37./225 - 169*sqrt(6)/1800,
                          -2./225 + sqrt(6)/75],
                         [37./225 + 169*sqrt(6)/1800, 11./45 - 7*sqrt(6) / 360,
                          -2./225 - sqrt(6)/75],
                         [4./9 - sqrt(6)/36, 4./9 + sqrt(6)/36, 1./9]])
        b = numpy.array([4./9-sqrt(6)/36, 4./9 + sqrt(6)/36, 1./9])
        c = numpy.array([2./5 - sqrt(6)/10, 2./5 + sqrt(6)/10, 1.0])
        super(Radau35, self).__init__(A, b, c)
