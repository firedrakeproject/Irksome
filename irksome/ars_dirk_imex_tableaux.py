import numpy as np

from .dirk_imex_tableaux import DIRK_IMEX

# Butcher tableau based on Ascher, Ruuth, and Spiteri Applied Numerical Mathematics 1997 (ARS)

# ARS tableau assume a zero first column of the implicit A matrix, so only the lower right s x s
# block is given for the implicit scheme. b and c are also of length s.

# Butcher tableau for s = 1
ars111A = np.array([[1.0]])
ars111A_hat = np.array([[1.0]])
ars111b = np.array([1.0])
ars111b_hat = np.array([1.0, 0.0])
ars111c = np.array([1.0])
ars111c_hat = np.array([0.0, 1.0])

ars121A = np.array([[1.0]])
ars121A_hat = np.array([[1.0]])
ars121b = np.array([1.0])
ars121b_hat = np.array([0.0, 1.0])
ars121c = np.array([1.0])
ars121c_hat = np.array([0.0, 1.0])

ars122A = np.array([[0.5]])
ars122A_hat = np.array([[0.5]])
ars122b = np.array([1.0])
ars122b_hat = np.array([0.0, 1.0])
ars122c = np.array([0.5])
ars122c_hat = np.array([0.0, 0.5])

# Butcher tableau for s = 2
gamma233 = (3 + np.sqrt(3))/6
ars233A = np.array([[gamma233, 0], [1 - 2*gamma233, gamma233]])
ars233A_hat = np.array([[gamma233, 0], [gamma233 - 1, 2*(1 - gamma233)]])
ars233b = np.array([0.5, 0.5])
ars233b_hat = np.array([0, 0.5, 0.5])
ars233c = np.array([gamma233, 1 - gamma233])
ars233c_hat = np.array([0, gamma233, 1 - gamma233])

gamma232 = (2 - np.sqrt(2)) / 2
delta232 = -2 * np.sqrt(2) / 3
ars232A = np.array([[gamma232, 0], [1 - gamma232, gamma232]])
ars232A_hat = np.array([[gamma232, 0], [delta232, 1 - delta232]])
ars232b = np.array([1 - gamma232, gamma232])
ars232b_hat = np.array([0, 1 - gamma232, gamma232])
ars232c = np.array([gamma232, 1.0])
ars232c_hat = np.array([0, gamma232, 1.0])

gamma222 = gamma232
delta222 = 1 - 1/(2*gamma222)
ars222A = np.array([[gamma222, 0], [1 - gamma222, gamma222]])
ars222A_hat = np.array([[gamma222, 0], [delta222, 1 - delta222]])
ars222b = np.array([1 - gamma222, gamma222])
ars222b_hat = np.array([delta222, 1 - delta222, 0])
ars222c = np.array([gamma222, 1.0])
ars222c_hat = np.array([0, gamma222, 1.0])

# Butcher tableau for s = 3
ars343A = np.array([[0.4358665215, 0, 0], [0.2820667392, 0.4358665215, 0], [1.208496649, -0.644363171, 0.4358665215]])
ars343A_hat = np.array([[0.4358665215, 0, 0], [0.3212788860, 0.3966543747, 0], [-0.105858296, 0.5529291479, 0.5529291479]])
ars343b = np.array([1.208496649, -0.644363171, 0.4358665215])
ars343b_hat = np.array([0, 1.208496649, -0.644363171, 0.4358665215])
ars343c = np.array([0.4358665215, 0.7179332608, 1])
ars343c_hat = np.array([0, 0.4358665215, 0.7179332608, 1.0])

# Butcher tableau for s = 4
ars443A = np.array([[1/2, 0, 0, 0],
                    [1/6, 1/2, 0, 0],
                    [-1/2, 1/2, 1/2, 0],
                    [3/2, -3/2, 1/2, 1/2]])
ars443A_hat = np.array([[1/2, 0, 0, 0],
                        [11/18, 1/18, 0, 0],
                        [5/6, -5/6, 1/2, 0],
                        [1/4, 7/4, 3/4, -7/4]])
ars443b = np.array([3/2, -3/2, 1/2, 1/2])
ars443b_hat = np.array([1/4, 7/4, 3/4, -7/4, 0])
ars443c = np.array([1/2, 2/3, 1/2, 1])
ars443c_hat = np.array([0, 1/2, 2/3, 1/2, 1])

ars_dict = {
    (1, 1, 1): (ars111A, ars111b, ars111c, ars111A_hat, ars111b_hat, ars111c_hat),
    (1, 2, 1): (ars121A, ars121b, ars121c, ars121A_hat, ars121b_hat, ars121c_hat),
    (1, 2, 2): (ars122A, ars122b, ars122c, ars122A_hat, ars122b_hat, ars122c_hat),
    (2, 2, 2): (ars222A, ars222b, ars222c, ars222A_hat, ars222b_hat, ars222c_hat),
    (2, 3, 2): (ars232A, ars232b, ars232c, ars232A_hat, ars232b_hat, ars232c_hat),
    (2, 3, 3): (ars233A, ars233b, ars233c, ars233A_hat, ars233b_hat, ars233c_hat),
    (3, 4, 3): (ars343A, ars343b, ars343c, ars343A_hat, ars343b_hat, ars343c_hat),
    (4, 4, 3): (ars443A, ars443b, ars443c, ars443A_hat, ars443b_hat, ars443c_hat)
}


class ARS_DIRK_IMEX(DIRK_IMEX):
    """Class to generate IMEX tableaux based on Ascher, Ruuth, and Spiteri (ARS). It has members

    :arg ns_imp: number of implicit stages
    :arg ns_exp: number of explicit stages
    :arg order: the (integer) former order of accuracy of the method
    """
    def __init__(self, ns_imp, ns_exp, order):
        try:
            A, b, c, A_hat, b_hat, c_hat = ars_dict[ns_imp, ns_exp, order]
        except KeyError:
            raise NotImplementedError("No ARS DIRK-IMEX method for that combination of implicit and explicit stages and order")

        # Expand A, b, c with assumed zeros in ARS tableaux
        A = self._pad_matrix(A, "lr")
        b = np.append(np.zeros(1), b)
        c = np.append(np.zeros(1), c)

        super(ARS_DIRK_IMEX, self).__init__(A, b, c, A_hat, b_hat, c_hat, order)
