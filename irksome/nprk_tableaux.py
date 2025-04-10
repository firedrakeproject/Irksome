"""Tableaux for non-linearly partitioned Runge-Kutta (NPRK) methods"""
from functools import cached_property

import numpy as np


def _check_tensor(tensor: dict, num_indices: int, name: str = "Tensor",
                  check_implicit: bool = False):
    """Checks Butcher tensor or weighting matrix"""

    if not isinstance(tensor, dict):
        raise TypeError(f"{name} must be a dictionary")

    for indices, value in tensor.items():
        if not isinstance(indices, tuple):
            raise TypeError(f"{name} keys must be tuples")
        if not len(indices) == num_indices:
            raise ValueError(f"{name} keys must have {num_indices} indices")
        if not all(isinstance(ind, int) for ind in indices):
            raise TypeError(f"{name} keys must be composed of integers")

        # Check Butcher tensor has the correct implicit structure
        if check_implicit:
            if indices[0] == 0:
                raise ValueError("i > 0 required, explicit first stage enforced")

            if not all(ind <= indices[0] for ind in indices[1:]):
                raise ValueError("j, k <= i required, cannot depend on future stages")
            if not indices.count(indices[0]) == 1:
                raise ValueError("j < i or k < i required, "
                                 + "must be implicit in exactly one argument")

        if not isinstance(value, float):
            raise TypeError(f"{name} values must be floats")


def _convert_tensor(tensor: dict) -> tuple[np.ndarray]:
    """Converts keys and values to two NumPy arrays for easier access"""

    num_entries = len(tensor)
    num_indices = len(tuple(tensor.keys())[0])

    tensor_indices = np.zeros((num_entries, num_indices), dtype=int)
    tensor_values = np.zeros(num_entries, dtype=float)

    for i, (key, value) in enumerate(tensor.items()):
        tensor_indices[i, :] = key
        tensor_values[i] = value

    if len(np.unique(tensor_indices, axis=0)) != tensor_indices.shape[0]:
        raise ValueError("All tensor indices must be unique")

    # Remove zeros
    zero_locs = np.where(np.abs(tensor_values) <= 1e-15)[0]
    tensor_indices = np.delete(tensor_indices, zero_locs, 0)
    tensor_values = np.delete(tensor_values, zero_locs, 0)

    # Sort by indices i, then j, then k
    sort_ord = np.lexsort(np.flip(tensor_indices, 1).T)
    tensor_indices = tensor_indices[sort_ord]
    tensor_values = tensor_values[sort_ord]

    return tensor_indices, tensor_values


class NPRKTableau:
    """Tableau for non-linearly partitioned Runge-Kutta (NPRK) methods.
    Since the Butcher tensors, a, and weight matrices, b, each have an extra dimension, they will
    typically be sparse and are stored accordingly.

    :arg a: Butcher tensor as a :class:`dict`. The keys are tuples of three integers
            (i,j,k) and the values are floats a_ijk.
    :arg b: Weighting matrix as a :class:`dict`. The keys are tuples of two integers (j,k) and
            the values are floats b_jk.
    :arg order: The order of accuracy of the scheme.
    :arg stiffly_accurate: Indicator the scheme is stiffly accurate. The weighting matrix is not
            needed and will be ignored if passed.
    """
    def __init__(self, a: dict[tuple[int], float], b: dict[tuple[int], float] = None,
                 order: int = None, stiffly_accurate: bool = False):

        if not stiffly_accurate and b is None:
            raise ValueError("Tableau must be stiffly accurate if b is not given")

        if not isinstance(order, int):
            raise TypeError("order must be an integer")
        self.order = order

        if not isinstance(stiffly_accurate, bool):
            raise TypeError("stiffly_accurate must be Boolean")
        self._stiffly_accurate = stiffly_accurate

        _check_tensor(a, 3, "Butcher tensor")
        self.a_indices, self.a_values = _convert_tensor(a)
        self.c_values = self._abscissae()

        if not stiffly_accurate:
            _check_tensor(b, 2, "Weighting matrix")
            self.b_indices, self.b_values = _convert_tensor(b)
        else:
            self.b_indices, self.b_values = None, None

    def _abscissae(self) -> np.ndarray:
        """Calculates abscissae c_i = sum_{j,k} a_ijk"""
        unique_i = np.unique(self.a_indices[:, 0])
        abscissae = np.zeros(unique_i.size, dtype=float)

        # index is array index, i is stage index
        for index, i in enumerate(unique_i):
            locs = np.where(self.a_indices[:, 0] == i)[0]
            abscissae[index] = np.sum(self.a_values[locs])

        return abscissae

    @cached_property
    def all_indices(self):
        """All indices (j,k) for which the operator F(Y_j,Y_k) needs to be evaluated"""
        if self.b_indices is None:
            pair_list = self.a_indices[:, 1:]
        else:
            pair_list = np.vstack((self.a_indices[:, 1:], self.b_indices))

        unique_pairs = np.unique(pair_list, axis=0)
        return unique_pairs

    @cached_property
    def _impl_locs(self):
        """List of entries corresponding to an implicit coefficient"""
        a_indices = self.a_indices
        mask = (a_indices[:, 0] == a_indices[:, 1]) | (a_indices[:, 0] == a_indices[:, 2])
        return mask

    @cached_property
    def impl_indices(self):
        """List of stage indices (j,k) for which the operator F(Y_j,Y_k) is evaluated
        during implicit solves"""
        pairs = self.a_indices[self._impl_locs, 1:]
        return pairs

    @cached_property
    def expl_indices(self):
        """List of indices (j,k) for which the operator F(Y_j,Y_k) is evaluated
        via an explicit solve"""
        impl_indices = self.impl_indices
        all_indices = self.all_indices

        # Determine which indices are not found via implicit solves
        mask_inv = np.array([np.any(np.all(impl_indices == row, axis=1))
                             for row in all_indices])
        pairs = all_indices[~mask_inv]

        pairs = np.unique(pairs, axis=0)

        return pairs

    @cached_property
    def impl_coeffs(self):
        """List of implicit coefficients for each stage"""
        coeffs = self.a_values[self._impl_locs]
        return coeffs

    @cached_property
    def impl_args(self):
        """List of which argument is explicit on each stage"""
        impl_indices = self.impl_indices
        args = np.array(impl_indices[:, 0] < impl_indices[:, 1], dtype=int)
        return args

    @cached_property
    def a_list(self):
        """List of a_ijk to sum over previous operator evaluations at each stage"""
        a_indices = self.a_indices
        a_values = self.a_values

        unique_i = np.unique(a_indices[:, 0])

        a_list_ = []

        for i in unique_i:
            # Filter out implicit part
            locs = np.where((a_indices[:, 0] == i)
                            & (a_indices[:, 1] != i)
                            & (a_indices[:, 2] != i))[0]
            # Store as tuple of jk, a_ijk (for fixed i)
            a_list_.append((a_indices[locs, 1:], a_values[locs]))

        return a_list_

    @cached_property
    def expl_solve_dict(self):
        """Dictionary of (j,k) indices corresponding to F(Y_j,Y_k) to be calculated
        at each stage"""
        expl_indices = self.expl_indices

        expl_solve_dict_ = {}

        for jk in expl_indices:
            i = np.min(jk)
            if i in expl_solve_dict_:
                expl_solve_dict_[i].append(jk)
            else:
                expl_solve_dict_[i] = [jk]

        return expl_solve_dict_

    @cached_property
    def num_stages(self):
        """Number of stages in tableau"""
        ns = np.max(self.a_indices[:, 0]) + 1
        return ns

    @cached_property
    def is_stiffly_accurate(self):
        """Indicator if tableau is stiffly accurate"""

        if not self._stiffly_accurate:
            a_indices = self.a_indices
            a_values = self.a_values
            b_indices = self.b_indices
            b_values = self.b_values

            i_max = np.max(a_indices[:, 0])
            locs = np.where(a_indices[:, 0] == i_max)[0]

            sa_check = a_indices[locs, 1:].shape == b_indices.shape
            sa_check *= a_values[locs].shape == b_values.shape

            if sa_check:
                sa_check *= np.all(a_indices[locs, 1:] == b_indices)
                sa_check *= np.allclose(a_values[locs], b_values)

            self._stiffly_accurate = bool(sa_check)

        return self._stiffly_accurate

    @cached_property
    def is_imex(self):
        """Indicator if the tableau is implicit in only one argument across the stages"""
        is_imex_ = len(np.unique(self.impl_args)) == 1
        return is_imex_

    @cached_property
    def is_seq_coupled(self):
        """Indicator if the tableau is sequentially coupled"""
        a_indices = self.a_indices
        is_sc_ = np.max(np.abs(a_indices[:, 1] - a_indices[:, 2])) == 1
        return is_sc_


# Midpoint and Crank-Nicholson (5.19)
mp_a = {(1, 1, 0): 0.5,
        (2, 1, 0): 0.0, (2, 1, 2): 0.5}
mp_b = {(1, 0): 0.0, (1, 2): 1.0}
mp_sa = False
mp_order = 2

cn_a = {(1, 1, 0): 0.5,
        (2, 1, 0): 0.5, (2, 1, 2): 0.5}
cn_b = {(1, 0): 0.5, (1, 2): 0.5}
cn_sa = False
cn_order = 2

# Second-order stiffly accurate scheme
stiff2_a = {(1, 0, 0): 0.5, (1, 1, 0): 0.5,
            (2, 0, 0): 0.5, (2, 1, 2): 0.5}
stiff2_b = None
stiff2_sa = True
stiff2_order = 2

# Third-order stiffly accurate scheme
stiff3_a = {(1, 0, 0): 1/12, (1, 1, 0): 1/4,
            (2, 0, 0): 1/12, (2, 1, 2): 1/4,
            (3, 1, 0): -1/2, (3, 1, 2): 5/4, (3, 3, 2): 1/4,
            (4, 1, 2): 3/4, (4, 3, 4): 1/4}
stiff3_b = None
stiff3_sa = True
stiff3_order = 3

imim_tableau_dict = {"mp": (mp_a, mp_b, mp_sa, mp_order),
                     "cn": (cn_a, cn_b, cn_sa, cn_order),
                     "stiff2": (stiff2_a, stiff2_b, stiff2_sa, stiff2_order),
                     "stiff3": (stiff3_a, stiff3_b, stiff3_sa, stiff3_order)}


class IMIMNPRKTableau(NPRKTableau):
    """Implements various IMIM NPRK tableaux.

    :arg name: Tableau name. Must be 'mp', 'cn', 'stiff2', or 'stiff3'"""

    def __init__(self, name: str = "mp"):
        try:
            a, b, stiffly_accurate, order = imim_tableau_dict[name]
        except KeyError:
            raise NotImplementedError("No IMIM NPRK tableau for that name. "
                                      f"Valid options are: {list(imim_tableau_dict.keys())}")

        super().__init__(a, b, order, stiffly_accurate)


imex_euler_a = {(1, 1, 0): 1.0}
imex_euler_b = {(1, 0): 1.0}
imex_euler_sa = True
imex_euler_order = 1

imex_mp_a = {(1, 1, 0): 0.5, (2, 1, 0): 0.5}
imex_mp_b = {(2, 1): 1.0}
imex_mp_sa = False
imex_mp_order = 2

imex_tableau_dict = {"1[21]": (imex_euler_a, imex_euler_b, imex_euler_sa, imex_euler_order),
                     "euler": (imex_euler_a, imex_euler_b, imex_euler_sa, imex_euler_order),
                     "2[31]": (imex_mp_a, imex_mp_b, imex_mp_sa, imex_mp_order),
                     "mp": (imex_mp_a, imex_mp_b, imex_mp_sa, imex_mp_order)}


class IMEXNPRKTableau(NPRKTableau):
    """Implements various IMEX NPRK tableaux.

    :arg name: Tableau name"""

    def __init__(self, name: str = "mp"):
        try:
            a, b, stiffly_accurate, order = imex_tableau_dict[name]
        except KeyError:
            raise NotImplementedError("No IMEX NPRK tableau for that name. "
                                      f"Valid options are: {list(imex_tableau_dict.keys())}")

        super().__init__(a, b, order, stiffly_accurate)
