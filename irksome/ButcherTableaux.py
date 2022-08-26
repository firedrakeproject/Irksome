import FIAT
import numpy
from numpy import vander, zeros
from numpy.linalg import solve


class ButcherTableau(object):
    """Top-level class representing a Butcher tableau encoding
       a Runge-Kutta method.  It has members

    :arg A: a 2d array containing the Butcher matrix
    :arg b: a 1d array giving weights assigned to each stage when
            computing the solution at time n+1.
    :arg btilde: If present, a 1d array giving weights for an embedded
            lower-order method (used in estimating temporal
            truncation error.)
    :arg c: a 1d array containing weights at which time-dependent
            terms are evaluated.
    :arg order: the (integer) formal order of accuracy of the method
    """
    def __init__(self, A, b, btilde, c, order):

        self.A = A
        self.b = b
        self.btilde = btilde
        self.c = c
        self.order = order

    @property
    def num_stages(self):
        """Return the number of stages the method has."""
        return len(self.b)

    @property
    def is_stiffly_accurate(self):
        """Determines whether the method is stiffly accurate."""
        res = zeros(self.num_stages)
        res[-1] = 1.0
        try:
            return numpy.allclose(res, solve(self.A.T, self.b))
        except numpy.linalg.LinAlgError:
            return False

    @property
    def is_explicit(self):
        A = self.A
        ns = self.num_stages
        for i in range(ns):
            for j in range(i, ns):
                if abs(A[i, j]) > 1.e-15:
                    return False
        return True

    @property
    def is_diagonally_implicit(self):
        A = self.A
        ns = self.num_stages
        for i in range(ns):
            for j in range(i+1, ns):
                if abs(A[i, j]) > 1.e-15:
                    return False
        return True

    @property
    def is_implicit(self):
        return not self.is_explicit

    @property
    def is_fully_implicit(self):
        return self.is_implicit and not self.is_diagonally_implicit
        
    def __str__(self):
        return str(self.__class__).split(".")[-1][:-2]+"()"


class CollocationButcherTableau(ButcherTableau):
    """When an RK method is based on collocation with point sets present
    in FIAT, we have a general formula for producing the Butcher tableau.

    :arg L: a one-dimensional class :class:`FIAT.FiniteElement`
            of Lagrange type -- the degrees of freedom must all be point
            evaluation.
    :arg order: the order of the resulting RK method.
    """
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

        Aexplicit = numpy.zeros((num_stages, num_stages))
        for i in range(num_stages):
            qpts_i = 1 + qpts * c[i]
            qwts_i = qwts * c[i]
            Lvals_i = L.tabulate(0, qpts_i)[0, ]
            Aexplicit[i, :] = Lvals_i @ qwts_i

        self.Aexplicit = Aexplicit

        V = vander(c, increasing=True)
        rhs = numpy.array([1.0/(s+1) for s in range(num_stages-1)] + [0])
        btilde = solve(V.T, rhs)

        super(CollocationButcherTableau, self).__init__(A, b, btilde, c, order)


class GaussLegendre(CollocationButcherTableau):
    """Collocation method based on the Gauss-Legendre points.
    The order of accuracy is 2 * `num_stages`.
    GL methods are A-stable, B-stable, and symplectic.

    :arg num_stages: The number of stages (1 or greater)
    """
    def __init__(self, num_stages):
        assert num_stages > 0
        U = FIAT.ufc_simplex(1)
        L = FIAT.GaussLegendre(U, num_stages - 1)
        super(GaussLegendre, self).__init__(L, 2 * num_stages)

    def __str__(self):
        return "GaussLegendre(%d)" % self.num_stages


class LobattoIIIA(CollocationButcherTableau):
    """Collocation method based on the Gauss-Lobatto points.
    The order of accuracy is 2 * `num_stages` - 2.
    LobattoIIIA methods are A-stable but not B- or L-stable.

    :arg num_stages: The number of stages (2 or greater)
    """
    def __init__(self, num_stages):
        assert num_stages > 1
        U = FIAT.ufc_simplex(1)
        L = FIAT.GaussLobattoLegendre(U, num_stages - 1)
        super(LobattoIIIA, self).__init__(L, 2 * num_stages - 2)

    def __str__(self):
        return "LobattoIIIA(%d)" % self.num_stages


class RadauIIA(CollocationButcherTableau):
    """Collocation method based on the Gauss-Radau points.
    The order of accuracy is 2 * `num_stages` - 1.
    RadauIIA methods are algebraically (hence B-) stable.

    :arg num_stages: The number of stages (2 or greater)
    """
    def __init__(self, num_stages):
        assert num_stages >= 1
        U = FIAT.ufc_simplex(1)
        L = FIAT.GaussRadau(U, num_stages - 1)
        super(RadauIIA, self).__init__(L, 2 * num_stages - 1)

    def __str__(self):
        return "RadauIIA(%d)" % self.num_stages


class BackwardEuler(RadauIIA):
    """The rock-solid first-order implicit method."""
    def __init__(self):
        super(BackwardEuler, self).__init__(1)

    def __str__(self):
        return ButcherTableau.__str__(self)


class LobattoIIIC(ButcherTableau):
    """Discontinuous collocation method based on the Lobatto points.
    The order of accuracy is 2 * `num_stages` - 2.
    LobattoIIIC methods are A-, L-, algebraically, and B- stable.

    :arg num_stages: The number of stages (2 or greater)
    """
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

    def __str__(self):
        return "LobattoIIIC(%d)" % self.num_stages


class PareschiRusso(ButcherTableau):
    """Second order, diagonally implicit, 2-stage.
    A-stable if x >= 1/4 and L-stable iff x = 1 plus/minus 1/sqrt(2)."""
    def __init__(self, x):
        self.x = x
        A = numpy.array([[x, 0.0], [1-2*x, x]])
        b = numpy.array([0.5, 0.5])
        c = numpy.array([x, 1-x])
        super(PareschiRusso, self).__init__(A, b, None, c, 2)

    def __str__(self):
        return "PareschiRusso(%f)" % self.x


class QinZhang(PareschiRusso):
    "Symplectic Pareschi-Russo DIRK"
    def __init__(self):
        super(QinZhang, self).__init__(0.25)

    def __str__(self):
        return "QinZhang()"
