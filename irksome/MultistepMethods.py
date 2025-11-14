import numpy as np


# BDF Methods

BDF1a = np.array([0.0, 1.0])
BDF1b = np.array([0.0, 1.0])
BDF2a = np.array([1.0 / 3.0, -4.0 / 3.0, 1.0])
BDF2b = np.array([0.0, 0.0, 2.0 / 3.0])
BDF3a = np.array([- 2.0 / 11.0, 9.0 / 11.0, -18.0 / 11.0, 1.0])
BDF3b = np.array([0.0, 0.0, 0.0, 6.0 / 11.0])

multistep_dict = {
    'BDF1': (BDF1a, BDF1b),
    'BDF2': (BDF2a, BDF2b),
    'BDF3': (BDF3a, BDF3b),
}
