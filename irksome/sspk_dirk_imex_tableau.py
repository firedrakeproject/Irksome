import numpy as np

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

sspk_dict = {
    (2, 2, 2, 2): (ssp2_222A, ssp2_222b, ssp2_222c, ssp2_222A_hat, ssp2_222b_hat, ssp2_222c_hat),
    (2, 3, 2, 2): (ssp2_322A, ssp2_322b, ssp2_322c, ssp2_322A_hat, ssp2_322b_hat, ssp2_322c_hat),
    (2, 3, 3, 2): (ssp2_332A, ssp2_332b, ssp2_332c, ssp2_332A_hat, ssp2_332b_hat, ssp2_332c_hat),
    (3, 3, 3, 2): (ssp3_332A, ssp3_332b, ssp3_332c, ssp3_332A_hat, ssp3_332b_hat, ssp3_332c_hat)
}
