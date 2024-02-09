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

        points = []
        for ell in L.dual.nodes:
            assert isinstance(ell, FIAT.functional.PointEvaluation)
            # Assert singleton point for each node.
            pt, = ell.get_point_dict().keys()
            points.append(pt[0])

        c = numpy.asarray(points)
        # GLL DOFs are ordered by increasing entity dimension!
        perm = numpy.argsort(c)
        c = c[perm]

        num_stages = len(c)

        Q = FIAT.make_quadrature(L.ref_el, 2*num_stages)
        qpts = Q.get_points()
        qwts = Q.get_weights()

        Lvals = L.tabulate(0, qpts)[0, ][perm]

        # integrates them all!
        b = Lvals @ qwts

        # now for A, which means we have to adjust the interval
        A = numpy.zeros((num_stages, num_stages))
        for i in range(num_stages):
            qpts_i = qpts * c[i]
            qwts_i = qwts * c[i]
            Lvals_i = L.tabulate(0, qpts_i)[0, ][perm]
            A[i, :] = Lvals_i @ qwts_i

        Aexplicit = numpy.zeros((num_stages, num_stages))
        for i in range(num_stages):
            qpts_i = 1 + qpts * c[i]
            qwts_i = qwts * c[i]
            Lvals_i = L.tabulate(0, qpts_i)[0, ][perm]
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


class Alexander(ButcherTableau):
    """ Third-order, diagonally implicit, 3-stage, L-stable scheme from
    Diagonally Implicit Runge-Kutta Methods for Stiff O.D.E.'s,
    R. Alexander, SINUM 14(6): 1006-1021, 1977."""
    def __init__(self):
        # Root of x^3 - 3x^2 +3x/2-1/6 between 1/6 and 1/2
        x = 0.43586652150845899942
        y = -1.5*x*x + 4*x - 0.25
        z = 1.5*x*x-5*x+1.25
        A = numpy.array([[x, 0.0, 0.0], [(1-x)/2.0, x, 0.0], [y, z, x]])
        b = numpy.array([y, z, x])
        c = numpy.array([x, (1+x)/2.0, 1])
        super(Alexander, self).__init__(A, b, None, c, 3)

    def __str__(self):
        return "Alexander()"

class WSODIRK432(ButcherTableau):
    def __init__(self):
        c = numpy.asarray(
            [0.01900072890, 0.78870323114, 0.41643499339, 1])
        b = numpy.asarray(
            [0.02343549374, −0.41207877888, 0.96661161281, 0.42203167233])
        A = numpy.asarray(
            [[0.01900072890, 0, 0, 0],
             [0.40434605601, 0.38435717512, 0, 0],
             [0.06487908412, −0.16389640295, 0.51545231222, 0],
             [0.02343549374, −0.41207877888, 0.96661161281, 0.42203167233]])
        super(WSODIRK432, self).__init__(A, b, None, c, 3)

    def __str__(self):
        return "WSIDIRK432()"

class WSODIRK433(ButcherTableau):
    def __init__(self):
        c = numpy.array(
            [0.13756543551, 0.80179011576, 2.33179673002, 1])
        b = numpy.array(
            [0.59761291500, −0.43420997584, −0.05305815322, 0.88965521406])
        A = numpy.array(
            [[0.13756543551, 0, 0, 0],
             [0.56695122794, 0.23483888782, 0, 0],
             [−1.08354072813, 2.96618223864, 0.44915521951, 0]
             [0.59761291500, −0.43420997584, −0.05305815322, 0.88965521406]])
        super(WSODIRK433, self).__init__(A, b, None, c, 3)

    def __str(self):
        return "WSODIRK433()"

    
class WSODIRK643(ButcherTableau):
    def __init__(self):
        c = numpy.array(
            [0.079672377876931, 0.464364648310935,
             1.348559241946724, 1.312664210308764,
             0.989469293495897, 1])
        b = numpy.array(
            [0.214823667785537, 0.536367363903245,
             0.154488125726409, −0.217748592703941,
             0.072226422925896, 0.239843012362853])
        A = numpy.array(
            [[0.079672377876931, 0, 0, 0, 0, 0],
             [0.328355391763968, 0.136009256546967, 0, 0, 0, 0],
             [−0.650772774016417, 1.742859063495349, 0.256472952467792, 0, 0, 0],
             [−0.714580550967259, 1.793745752775934, −0.078254785672497, 0.311753794172585, 0, 0],
             [−1.120092779092918, 1.983452339867353, 3.117393885836001, −3.761930177913743, 0.770646024799205, 0],
             [0.214823667785537, 0.536367363903245,
             0.154488125726409, −0.217748592703941,
             0.072226422925896, 0.239843012362853]])
        super(WSODIRK643, self).__init__(A, b, None, c, 4)

    def __str(self):
        return "WSODIRK643()"
    
class WSODIRK744(ButcherTableau):
    """From Biswas et al"""
    def __init__(self):
        c = numpy.asarray(
            [1.290066345260422e-01, 4.492833135308985e-01,
             9.919659086525534e-03, 1.230475897454758e+00,
             2.978701803613543e+00, 1.247908693583052e+00,
             1.000000000000000e+00])
        b = numpy.asarray(
            [2.387938238483883e-01, 4.762495400483653e-01,
             1.233935151213300e-02, 6.011995982693821e-02,
             6.553618225489034e-05, -1.270730910442124e-01,
             3.395048796261326e-01])
        A = numpy.array([[1.290066345260422e-01, 0, 0, 0, 0, 0, 0],
            [3.315354455306989e-01, 1.177478680001996e-01, 0, 0, 0, 0, 0],
            [-8.009819642882672e-02, -2.408450965101765e-03, 9.242630648045402e-02, 0, 0, 0, 0],
            [-1.730636616639455e+00, 1.513225984674677e+00, 1.221258626309848e+00, 2.266279031096887e-01, 0, 0, 0],
            [1.475353790517696e-01, 3.618481772236499e-01, -5.603544220240282e-01, 2.455453653222619e+00, 5.742190161395324e-01, 0, 0],
                         [2.099717815888321e-01, 7.120237463672882e-01, -2.012023940726332e-02, -1.913828539529156e-02, -5.556044541810300e-03, 3.707277349712966e-01, 0],
            [2.387938238483883e-01, 4.762495400483653e-01, 1.233935151213300e-02, 6.011995982693821e-02, 6.553618225489034e-05, -1.270730910442124e-01, 3.395048796261326e-01]])
        super(WSODIRK744, self).__init__(A, b, None, c, 4)

    def __str__(self):
        return "WSIDIRK744()"
    

class WSDIRK1254():
    def __init__(self):
        c = numpy.array(
            [2.345371908646273e-01, 7.425871511958302e-01,
             3.296674204078279e-02, 7.379564717201322e-01,
             2.376643917109970e-01, 1.750238160341377e+00,
             2.990308150015702e+00, 2.882138003112822e+00,
             2.914399924907188e+00, 2.573507348677332e+00,
             3.567266961364713e+00, 1.000000000000000e+00])
        b = numpy.array(
            [-7.433675378768276e-01, 1.490594423766965e-01,
             -2.042884056742363e-02, 8.565329438087443e-04,
             1.357261590983184e+00, 2.067512027776675e-03,
             9.836884265759428e-02, -1.357936974507222e-02,
             -5.428992174996300e-02, -3.803299038293005e-02,
             -9.150525836295019e-03, 2.712352651694511e-01])
        A = numpy.array(
            [[2.345371908646273e-01,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [6.874344413888787e-01, 5.515270980695153e-02,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-1.183552669539587e-01, 5.463563002913454e-03,
              1.458584459918280e-01,
              0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-1.832235204042292e-01, 5.269029412008775e-02,
              8.203685085133529e-01, 4.812118949092085e-02,
              0, 0, 0, 0, 0, 0, 0, 0],
             [9.941572060659400e-02, 4.977904930055774e-03,
              5.414758174284321e-02, -1.666571741820749e-03,
              8.078975617332473e-02,
              0, 0, 0, 0, 0, 0, 0],
             [-9.896614721582678e-01, 2.860682690577833e+00,
              -1.236119341063179e+00, 2.130219523351530e+00,
              -1.260655031676537e+00, 2.457717913099987e-01,
              0, 0, 0, 0, 0, 0],
             [-5.656238413439102e-02, 1.661985685769353e-01,
              6.464600922362508e-01, 6.608854962269927e-01,
              3.736054198873429e-01, 6.294456964407685e-01,
              5.702752607818027e-01,
              0, 0, 0, 0, 0],
             [8.048962104724392e-01, -6.232034990249100e-02,
              5.737234603323347e-01, -9.613723511489970e-02,
              5.524106361737929e-01, 5.961002486833255e-01,
              1.978411600659203e-01, 3.156238724024008e-01,
              0, 0, 0, 0],
             [-1.606381759216300e-01, 6.833397073337708e-01,
              4.734578665308685e-01, 8.037708984872738e-01,
              -1.094498069459834e-02, 6.151263362711297e-01,
              3.908946848682723e-01, 8.966103265353116e-02,
              2.973255537857041e-02,
              0, 0, 0],
             [7.074283235644631e-01, 4.392037300952482e-01,
              -3.623592480237268e-02, 7.189990308645932e-04,
              5.820968279166545e-01, 3.302003177175218e-01,
              -2.394564021215881e-01, -7.540283547997615e-03,
              1.702137469523672e-01, 6.268780138721711e-01,
              0, 0],
             [1.361197981133694e-01, -7.486549901902831e-01,
              1.893908350024949e+00, 3.940485196730028e-01,
              6.240233526545023e-02, 7.511983862200027e-01,
              -5.283465265730526e-01, -1.661625677872943e+00,
              9.998723833190827e-01, 1.377776742457387e+00,
              8.905676409277480e-01, 0],
             [-7.433675378768276e-01, 1.490594423766965e-01,
              -2.042884056742363e-02, 8.565329438087443e-04,
              1.357261590983184e+00, 2.067512027776675e-03,
              9.836884265759428e-02, -1.357936974507222e-02,
              -5.428992174996300e-02, -3.803299038293005e-02,
              -9.150525836295019e-03, 2.712352651694511e-01]])

        super(WSODIRK1254, self).__init__(A, b, None, c, 5)

    def __str__(self):
        return "WSODIRK1254()"


class WSODIRK1255(ButcherTableau):
    def __init__(self):
        c = numpy.array(
            [4.113473525867655e-02, 2.269850660400232e-01,
             6.222969192243949e-01, 1.377989449231234e+00,
             1.259841986970257e+00, 1.228350442796143e+00,
             1.269855051265635e+00, 2.496200652601413e+00,
             2.783820705331141e+00, 3.337101417632813e+00,
             4.173423133876636e+00, 1.000000000000000e+00])
        b = numpy.array(
            [1.207394392845339e-02, 5.187080074649261e-01,
             1.121304244847239e-01, -4.959806334780896e-03,
             -1.345031364651444e+00, 3.398828703760807e-01,
             8.159251531671077e-01, -2.640104266439604e-03,
             1.439060901763520e-02, -6.556567796749947e-03,
             6.548135446843367e-04, 5.454220210658036e-01])
        A = numpy.array(
            [[4.113473525867655e-02,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1.603459327727949e-01, 6.663913326722831e-02,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-3.424389044264752e-01, 8.658006324816373e-01,
              9.893519116923277e-02,
              0, 0, 0, 0, 0, 0, 0, 0, 0],
             [9.437182028870806e+00, -1.088783359642350e+01,
              2.644025436733866e+00, 1.846155800500574e-01,
              0, 0, 0, 0, 0, 0, 0, 0],
             [-3.425409029430815e-01, 5.172239272544332e-01,
              9.163589909678043e-01, 5.225142808845742e-02,
              1.165485436026433e-01,
              0, 0, 0, 0, 0, 0, 0],
             [-2.094441177460360e+00, 2.577655753533404e+00,
              5.704405293326313e-01, 1.213637180023516e-01,
              -4.752289775376601e-01, 5.285605969257756e-01,
              0, 0, 0, 0, 0, 0],
             [3.391631788320480e-01, -2.797427027028997e-01,
              1.039483063369094e+00, 5.978770926212172e-02,
              -2.132900327070380e-01, 8.344318363436753e-02,
              2.410106515779412e-01,
              0, 0, 0, 0, 0],
             [5.904282488642163e+00, 3.171195765985073e+00,
              -1.236822836316587e+01, -4.989519066913001e-01,
              2.160529620826442e+00, 1.916104322021480e+00,
              1.988059486291180e+00, 2.232092386922440e-01,
              0, 0, 0, 0],
             [4.616443509508975e-01, -1.933433560549238e-01,
              -1.212541486279519e-01, 6.662362039716674e-02,
              4.254912950625259e-01, 7.856131647013712e-01,
              8.369551389357689e-01, 1.604780447895926e-01,
              3.616125951766939e-01,
              0, 0, 0],
             [-7.087669749878204e-01, 6.466527094491541e-01,
              4.758821526542215e-01, -2.570518451375722e-01,
              1.123185062554392e+00, 5.546921612875290e-01,
              3.192424333237050e-01, 3.612077612576969e-01,
              5.866779836068974e-01, 2.353799736246102e-01,
              0, 0],
             [4.264162484855930e-01, 1.322816663477840e+00,
              4.245673729758231e-01, -2.530402764527700e+00,
              -7.822016897497742e-02, 1.054463080605071e+00,
              4.645590541391895e-01, 1.145097379521439e+00,
              4.301337846893282e-01, 1.499513057076809e+00,
              1.447942640822165e-02, 0],
             [1.207394392845339e-02, 5.187080074649261e-01,
              1.121304244847239e-01, -4.959806334780896e-03,
              -1.345031364651444e+00, 3.398828703760807e-01,
              8.159251531671077e-01, -2.640104266439604e-03,
              1.439060901763520e-02, -6.556567796749947e-03,
              6.548135446843367e-04, 5.454220210658036e-01]])

        super(WSODIRK1255, self).__init__(A, b, None, c, 5)

    def __str__(self):
        return "WSODIRK1255()"
