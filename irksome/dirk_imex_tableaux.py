from .ButcherTableaux import ButcherTableau
import numpy as np

# For the implicit scheme, the full Butcher Table is given as A, b, c.

# For the explicit scheme, the full b_hat and c_hat are given, but (to
# avoid a lot of offset-by-ones in the code we store only the
# lower-left ns x ns block of A_hat

# IMEX Butcher tableau for 1 stage
imex111A = np.array([[1.0]])
imex111A_hat = np.array([[1.0]])
imex111b = np.array([1.0])
imex111b_hat = np.array([1.0, 0.0])
imex111c = np.array([1.0])
imex111c_hat = np.array([0.0, 1.0])


# IMEX Butcher tableau for s = 2
gamma = (2 - np.sqrt(2)) / 2
delta = -2 * np.sqrt(2) / 3
imex232A = np.array([[gamma, 0], [1 - gamma, gamma]])
imex232A_hat = np.array([[gamma, 0], [delta, 1 - delta]])
imex232b = np.array([1 - gamma, gamma])
imex232b_hat = np.array([0, 1 - gamma, gamma])
imex232c = np.array([gamma, 1.0])
imex232c_hat = np.array([0, gamma, 1.0])

# IMEX Butcher tableau for 3 stages
imex343A = np.array([[0.4358665215, 0, 0], [0.2820667392, 0.4358665215, 0], [1.208496649, -0.644363171, 0.4358665215]])
imex343A_hat = np.array([[0.4358665215, 0, 0], [0.3212788860, 0.3966543747, 0], [-0.105858296, 0.5529291479, 0.5529291479]])
imex343b = np.array([1.208496649, -0.644363171, 0.4358665215])
imex343b_hat = np.array([0, 1.208496649, -0.644363171, 0.4358665215])
imex343c = np.array([0.4358665215, 0.7179332608, 1])
imex343c_hat = np.array([0, 0.4358665215, 0.7179332608, 1.0])


# IMEX Butcher tableau for 4 stages
imex443A = np.array([[1/2, 0, 0, 0],
                     [1/6, 1/2, 0, 0],
                     [-1/2, 1/2, 1/2, 0],
                     [3/2, -3/2, 1/2, 1/2]])
imex443A_hat = np.array([[1/2, 0, 0, 0],
                         [11/18, 1/18, 0, 0],
                         [5/6, -5/6, 1/2, 0],
                         [1/4, 7/4, 3/4, -7/4]])
imex443b = np.array([3/2, -3/2, 1/2, 1/2])
imex443b_hat = np.array([1/4, 7/4, 3/4, -7/4, 0])
imex443c = np.array([1/2, 2/3, 1/2, 1])
imex443c_hat = np.array([0, 1/2, 2/3, 1/2, 1])

dirk_imex_dict = {
    (1, 1, 1): (imex111A, imex111b, imex111c, imex111A_hat, imex111b_hat, imex111c_hat),
    (2, 3, 2): (imex232A, imex232b, imex232c, imex232A_hat, imex232b_hat, imex232c_hat),
    (3, 4, 3): (imex343A, imex343b, imex343c, imex343A_hat, imex343b_hat, imex343c_hat),
    (4, 4, 3): (imex443A, imex443b, imex443c, imex443A_hat, imex443b_hat, imex443c_hat)
}


class DIRK_IMEX(ButcherTableau):
    def __init__(self, ns_imp, ns_exp, order):
        try:
            A, b, c, A_hat, b_hat, c_hat = dirk_imex_dict[ns_imp, ns_exp, order]
        except KeyError:
            raise NotImplementedError("No DIRK-IMEX method for that combination of implicit and explicit stages and order")
        self.order = order
        super(DIRK_IMEX, self).__init__(A, b, None, c, order, None, None)
        self.A_hat = A_hat
        self.b_hat = b_hat
        self.c_hat = c_hat
        self.is_dirk_imex = True  # Mark this as a DIRK-IMEX scheme
