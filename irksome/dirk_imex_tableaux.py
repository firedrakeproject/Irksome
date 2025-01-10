from .ButcherTableaux import ButcherTableau
import numpy as np


# IMEX Butcher tableau using numpy arrays
class IMEXEuler(ButcherTableau):
    def __init__(self):
        A = np.array([[1.0]])    # Implicit matrix (for implicit part)
        A_hat = np.array([[0.0]])  # Explicit matrix (for explicit part)
        b = np.array([1.0])      # Implicit weights
        b_hat = np.array([1.0])  # Explicit weights
        c = np.array([1.0])      # Time steps for implicit part
        c_hat = np.array([0.0])  # Time steps for explicit part
        order = 1                  # First-order method (Euler)
        embedded_order = None      # set to None
        gamma0 = None              # set to None
        super().__init__(A, b, b_hat, c, order, embedded_order, gamma0)
        self.A_hat = A_hat
        self.b_hat = b_hat
        self.c_hat = c_hat
        self.is_imex = True  # Mark this as an IMEX scheme


# IMEX Butcher tableau for s = 2
class IMEX2(ButcherTableau):
    def __init__(self):
        # Parameters for the s = 2 method
        gamma = (2 - np.sqrt(2)) / 2
        delta = -2 * np.sqrt(2) / 3

        # Implicit and explicit coefficients
        A = np.array([[gamma, 0], [1 - gamma, gamma]])   # Implicit matrix (A)
        A_hat = np.array([[gamma, 0], [delta, 1 - delta]])  # Explicit matrix (A_hat)
        b = np.array([1 - gamma, gamma])     # Implicit weights
        b_hat = np.array([0, 1 - gamma, gamma])     # Explicit weights (b_hat)
        c = np.array([gamma, 1.0])     # Time steps for implicit part (c)
        c_hat = np.array([0, gamma, 1.0])    # Time steps for explicit part (c_hat)

        # The method order is 2
        order = 2
        embedded_order = None  # set to None
        gamma0 = None  # set to None
        btilde = None

        super().__init__(A, b, btilde, c, order, embedded_order, gamma0)
        self.A_hat = A_hat
        self.b_hat = b_hat
        self.c_hat = c_hat
        self.is_imex = True  # Mark this as an IMEX scheme


# IMEX Butcher tableau for s = 3
class IMEX3(ButcherTableau):
    def __init__(self):
        A = np.array([[0.4358665215, 0, 0], [0.2820667392, 0.4358665215, 0], [1.208496649, -0.644363171, 0.4358665215]])   # Implicit matrix (A)
        A_hat = np.array([[0.4358665215, 0, 0], [0.3212788860, 0.3966543747, 0], [-0.105858296, 0.5529291479, 0.5529291479]])  # Explicit matrix (A_hat)
        b = np.array([1.208496649, -0.644363171, 0.4358665215])     # Implicit weights
        b_hat = np.array([0, 1.208496649, -0.644363171, 0.4358665215])     # Explicit weights (b_hat)
        c = np.array([0.4358665215, 0.7179332608, 1])     # Time steps for implicit part (c)
        c_hat = np.array([0, 0.4358665215, 0.7179332608, 1.0])    # Time steps for explicit part (c_hat)

        # The method order is 3
        order = 3
        embedded_order = None  # set to None
        gamma0 = None  # set to None
        btilde = None

        super().__init__(A, b, btilde, c, order, embedded_order, gamma0)
        self.A_hat = A_hat
        self.b_hat = b_hat
        self.c_hat = c_hat
        self.is_imex = True  # Mark this as an IMEX scheme


# IMEX Butcher tableau for s = 4
class IMEX4(ButcherTableau):
    def __init__(self):
        A = np.array([[1/2, 0, 0, 0],
                      [1/6, 1/2, 0, 0],
                      [-1/2, 1/2, 1/2, 0],
                      [3/2, -3/2, 1/2, 1/2]])  # Corrected A matrix definition
        A_hat = np.array([[1/2, 0, 0, 0],
                          [11/18, 1/18, 0, 0],
                          [5/6, -5/6, 1/2, 0],
                          [1/4, 7/4, 3/4, -7/4]])  # Explicit matrix (A_hat)
        b = np.array([3/2, -3/2, 1/2, 1/2])  # Implicit weights
        b_hat = np.array([1/4, 7/4, 3/4, -7/4, 0])  # Explicit weights (b_hat)
        c = np.array([1/2, 2/3, 1/2, 1])  # Time steps for implicit part (c)
        c_hat = np.array([0, 1/2, 2/3, 1/2, 1])  # Time steps for explicit part (c_hat)

        order = 3
        embedded_order = None
        gamma0 = None
        btilde = None

        super().__init__(A, b, btilde, c, order, embedded_order, gamma0)
        self.A_hat = A_hat
        self.b_hat = b_hat
        self.c_hat = c_hat
        self.is_imex = True  # Mark this as an IMEX scheme
