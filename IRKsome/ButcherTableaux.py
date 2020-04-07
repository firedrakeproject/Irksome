import FIAT
import numpy
from numpy import sqrt, vander


class ButcherTableau(object):
    def __init__(self, A, b, btilde, c, order):
        self.A = A
        self.b = b
        self.btilde = btilde
        self.c = c
        self.order = order

    @property
    def num_stages(self):
        return len(self.b)


class BackwardEuler(ButcherTableau):
    def __init__(self):
        A = numpy.array([[1.0]])
        b = numpy.array([1.0])
        c = numpy.array([1.0])
        super(BackwardEuler, self).__init__(A, b, None, c, 1)


class CollocationButcherTableau(ButcherTableau):
    def __init__(self, L, order):
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

        V = vander(c, increasing=True)
        rhs = numpy.array([1.0/(s+1) for s in range(num_stages-1)] + [0])
        btilde = numpy.linalg.solve(V.T, rhs)
            
        super(CollocationButcherTableau, self).__init__(A, b, btilde, c, order)


class GaussLegendre(CollocationButcherTableau):
    def __init__(self, num_stages):
        assert num_stages > 0
        U = FIAT.ufc_simplex(1)
        L = FIAT.GaussLegendre(U, num_stages - 1)
        super(GaussLegendre, self).__init__(L, 2 * num_stages)


class LobattoIIIA(CollocationButcherTableau):
    def __init__(self, num_stages):
        assert num_stages > 1
        U = FIAT.ufc_simplex(1)
        L = FIAT.GaussLobattoLegendre(U, num_stages - 1)
        super(LobattoIIIA, self).__init__(L, 2 * num_stages - 2)


class Radau23(ButcherTableau):
    def __init__(self):
        A = numpy.array([[5./12, -1./12], [3./4, 1./4]])
        b = numpy.array([3./4, 1./4])
        c = numpy.array([1./3, 1.])
        super(Radau23, self).__init__(A, b, None, c, 3)


class Radau35(ButcherTableau):
    def __init__(self):
        A = numpy.array([[11./45 - 7*sqrt(6)/360, 37./225 - 169*sqrt(6)/1800,
                          -2./225 + sqrt(6)/75],
                         [37./225 + 169*sqrt(6)/1800, 11./45 - 7*sqrt(6) / 360,
                          -2./225 - sqrt(6)/75],
                         [4./9 - sqrt(6)/36, 4./9 + sqrt(6)/36, 1./9]])
        b = numpy.array([4./9-sqrt(6)/36, 4./9 + sqrt(6)/36, 1./9])
        c = numpy.array([2./5 - sqrt(6)/10, 2./5 + sqrt(6)/10, 1.0])
        super(Radau35, self).__init__(A, b, None, c, 5)


class LobattoIIIC(ButcherTableau):
    def __init__(self, num_stages):
        assert num_stages > 1
        # mooch the b and c from IIIA
        IIIA = LobattoIIIA(num_stages)
        b = IIIA.b
        c = IIIA.c

        A = numpy.zeros((num_stages, num_stages))
        for i in range(num_stages):
            A[i, 0] = b[0]
        for j in range(num_stages):
            A[-1, j] = b[j]

        mat = numpy.vander(c[1:], increasing=True).T
        for i in range(num_stages-1):
            rhs = numpy.array([(c[i]**(k+1))/(k+1) - b[0] * c[0]**k
                               for k in range(num_stages-1)])
            A[i, 1:] = numpy.linalg.solve(mat, rhs)

        super(LobattoIIIC, self).__init__(A, b, None, c, 2 * num_stages - 2)


class PareschiRusso(ButcherTableau):
    """Second order, diagonally implicit, 2-stage.  
    A-stable if x >= 1/4 and L-stable iff x = 1 plus/minus 1/sqrt(2)."""
    def __init__(self, x):
        A = numpy.array([[x, 0.0], [1-2*x, x]])
        b = numpy.array([0.5, 0.5])
        c = numpy.array([x, 1-x])
        super(PareschiRusso, self).__init__(A, b, None, c, 2)


class QinZhang(PareschiRusso):
    "Symplectic Pareschi-Russo DIRK"
    def __init__(self):
        super(QinZhang, self).__init__(0.25)

