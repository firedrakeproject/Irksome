import numpy as np

from .ButcherTableaux import ButcherTableau
from .dirk_imex_tableaux import DIRK_IMEX

# Butcher tableau based on schemes in Pareschi and Russo Journal of Scientific Computing 2005

# Butcher tableau for k = 2
gamma = 1 - 1/np.sqrt(2)
ssp2_222A = np.array([[gamma, 0], [1 - 2*gamma, gamma]])
ssp2_222A_hat = np.array([[1.0]])
ssp2_222b = np.array([0.5, 0.5])
ssp2_222b_hat = np.array([0.5, 0.5])
ssp2_222c = np.array([gamma, 1 - gamma])
ssp2_222c_hat = np.array([0, 1.0])

ssp2_322A = np.array([[0.5, 0, 0], [-0.5, 0.5, 0], [0, 0.5, 0.5]])
ssp2_322A_hat = np.array([[0, 0], [0, 1.0]])
ssp2_322b = np.array([0, 0.5, 0.5])
ssp2_322b_hat = np.array([0, 0.5, 0.5])
ssp2_322c = np.array([0.5, 0, 1.0])
ssp2_322c_hat = np.array([0, 0, 1.0])

ssp2_332A = np.array([[1/4., 0, 0], [0, 1/4., 0], [1/3., 1/3., 1/3.]])
ssp2_332A_hat = np.array([[1/2., 0], [1/2., 1/2.]])
ssp2_332b = np.array([1/3., 1/3., 1/3.])
ssp2_332b_hat = np.array([1/3., 1/3., 1/3.])
ssp2_332c = np.array([1/4., 1/4., 1.0])
ssp2_332c_hat = np.array([0, 1/2., 1.0])

# Butcher tableau for k = 3
ssp3_332A = np.array([[gamma, 0, 0], [1 - 2*gamma, gamma, 0], [1/2. - gamma, 0, gamma]])
ssp3_332A_hat = np.array([[1.0, 0], [1/4., 1/4.]])
ssp3_332b = np.array([1/6., 1/6., 2/3.])
ssp3_332b_hat = np.array([1/6., 1/6., 2/3.])
ssp3_332c = np.array([gamma, 1 - gamma, 1/2.])
ssp3_332c_hat = np.array([0, 1.0, 1/2.])

sspk_imex_dict = {
    (2, 2, 2, 2): (ssp2_222A, ssp2_222b, ssp2_222c, ssp2_222A_hat, ssp2_222b_hat, ssp2_222c_hat),
    (2, 3, 2, 2): (ssp2_322A, ssp2_322b, ssp2_322c, ssp2_322A_hat, ssp2_322b_hat, ssp2_322c_hat),
    (2, 3, 3, 2): (ssp2_332A, ssp2_332b, ssp2_332c, ssp2_332A_hat, ssp2_332b_hat, ssp2_332c_hat),
    (3, 3, 3, 2): (ssp3_332A, ssp3_332b, ssp3_332c, ssp3_332A_hat, ssp3_332b_hat, ssp3_332c_hat)
}


ssp_dict = {
    (2, 2): (ssp2_222A_hat, ssp2_222b_hat, ssp2_222c_hat),
    (2, 3): (ssp2_332A_hat, ssp2_332b_hat, ssp2_332c_hat),
    (3, 3): (ssp3_332A_hat, ssp3_332b_hat, ssp3_332c_hat)
}


class SSPButcherTableau(ButcherTableau):
    """Class used to generate tableau for strong stability preserving (SSP) schemes. It has members

    :arg ns: number of stages
    :arg order: the (integer) formal order of accuracy of the method"""
    def __init__(self, order, ns):
        try:
            A, b, c = ssp_dict[order, ns]
        except KeyError:
            raise NotImplementedError("No SSP method for that combination of order and number of stages")

        # Zero pad to match expectations of ExplicitTimeStepper
        A_ = np.zeros((A.shape[0]+1, A.shape[1]+1), dtype=A.dtype)
        A_[1:, :-1] = A

        super(SSPButcherTableau, self).__init__(A_, b, None, c, order, None, None)


class SSPK_DIRK_IMEX(DIRK_IMEX):
    """Class to generate IMEX tableaux based on Pareschi and Russo. It has members

    :arg ssp_order: order of ssp scheme
    :arg ns_imp: number of implicit stages
    :arg ns_exp: number of explicit stages
    :arg order: the (integer) formal order of accuracy of the method"""
    def __init__(self, ssp_order, ns_imp, ns_exp, order):
        try:
            A, b, c, A_hat, b_hat, c_hat = sspk_imex_dict[ssp_order, ns_imp, ns_exp, order]
        except KeyError:
            raise NotImplementedError("No SSPk DIRK-IMEX method for that combination of SSP order, implicit and explicit stages, and IMEX order")

        super(SSPK_DIRK_IMEX, self).__init__(A, b, c, A_hat, b_hat, c_hat, order)
