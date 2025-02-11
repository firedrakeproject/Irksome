from .ButcherTableaux import ButcherTableau
from .ars_dirk_imex_tableaux import ars_dict
from .sspk_dirk_imex_tableau import sspk_dict
import numpy as np


class DIRK_IMEX(ButcherTableau):
    """Top-level class representing a pair of Butcher tableau encoding an implicit-explicit
    additive Runge-Kutta method. Since the explicit Butcher matrix is strictly lower triangular,
    only the lower-left (ns - 1)x(ns - 1) block is given. However, the full b_hat and c_hat are
    given. It has members

    :arg A: a 2d array containing the implicit Butcher matrix
    :arg b: a 1d array giving weights assigned to each implicit stage when
            computing the solution at time n+1.
    :arg c: a 1d array containing weights at which time-dependent
            implicit terms are evaluated.
    :arg A_hat: a 2d array containing the explicit Butcher matrix (lower-left block only)
    :arg b_hat: a 1d array giving weights assigned to each explicit stage when
            computing the solution at time n+1.
    :arg c_hat: a 1d array containing weights at which time-dependent
            explicit terms are evaluated.
    :arg order: the (integer) formal order of accuracy of the method
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, A_hat: np.ndarray,
                 b_hat: np.ndarray, c_hat: np.ndarray, order: int = None):

        # Number of stages
        ns = A.shape[0]
        assert ns == A.shape[1], "A must be square"
        assert A_hat.shape == (ns - 1, ns - 1), "A_hat must have one fewer row and column than A"
        assert ns == len(b) == len(b_hat), \
            "b and b_hat must have the same length as the number of stages"
        assert ns == len(c) == len(c_hat), \
            "c and c_hat must have the same length as the number of stages"

        super().__init__(A, b, None, c, order, None, None)
        self.A_hat = self._pad_matrix(A_hat, "ll")
        self.b_hat = b_hat
        self.c_hat = c_hat
        self.is_dirk_imex = True  # Mark this as a DIRK-IMEX scheme

    @staticmethod
    def _pad_matrix(mat: np.ndarray, loc: str):
        """Zero pads a matrix"""
        n = mat.shape[0]
        assert n == mat.shape[1], "Matrix must be square"
        padded = np.zeros((n+1, n+1), dtype=mat.dtype)

        if loc == "ll":
            # Lower left corner
            padded[1:, :-1] = mat
        elif loc == "lr":
            # Lower right corner
            padded[1:, 1:] = mat
        else:
            raise ValueError("Location must be ll (lower left) or lr (lower right)")

        return padded


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


class SSPK_DIRK_IMEX(DIRK_IMEX):
    """Class to generate IMEX tableaux based on Pareschi and Russo. It has members

    :arg ssp_order: order of ssp scheme
    :arg ns_imp: number of implicit stages
    :arg ns_exp: number of explicit stages
    :arg order: the (integer) formal order of accuracy of the method"""
    def __init__(self, ssp_order, ns_imp, ns_exp, order):
        try:
            A, b, c, A_hat, b_hat, c_hat = sspk_dict[ssp_order, ns_imp, ns_exp, order]
        except KeyError:
            raise NotImplementedError("No SSPk DIRK-IMEX method for that combination of SSP order, implicit and explicit stages, and IMEX order")

        super(SSPK_DIRK_IMEX, self).__init__(A, b, c, A_hat, b_hat, c_hat, order)
