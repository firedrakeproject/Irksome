from .backend import get_backend
import numpy
import ufl
from ufl.algorithms.analysis import extract_type
from ufl.classes import Variable
from ufl import as_tensor, replace as ufl_replace

import FIAT

from .ufl.deriv import TimeDerivative
from .ufl.lag import lag_label


def dot(A, B):
    return numpy.tensordot(A, B, (-1, 0))


def reshape(expr, shape):
    return numpy.reshape([expr[i] for i in numpy.ndindex(expr.ufl_shape)], shape)


def flatten_dats(dats):
    from pyop2.types import MixedDat
    flat_dat = []
    for dat in dats:
        if isinstance(dat, (tuple, list, MixedDat)):
            flat_dat.extend(dat)
        else:
            flat_dat.append(dat)
    return MixedDat(flat_dat)


def get_stage_space(V, num_stages, backend="firedrake"):
    backend_cls = get_backend(backend)
    return backend_cls.get_stage_space(V, num_stages)


def split_stages(V, stages):
    """Reconstruct the stages as a list of Function(V)"""
    num_fields = len(V)
    if num_fields == 1:
        return stages.subfunctions

    stages_np = reshape(stages, (-1, *V.value_shape))
    ks = [as_tensor(stages_np[i]) for i in range(stages_np.shape[0])]
    return ks


def extract_timedep_arguments(F, u0):
    """Return both arguments if ``F`` is a bilinear form, otherwise
    return the unique argument and ``u0``.
    """
    try:
        v, u = F.arguments()
    except ValueError:
        v, = F.arguments()
        u = u0
    return v, u


def fields_to_components(V, fields):
    """
    Returns the scalar component indices corresponding to the possibly
    tensor-valued subspaces of a mixed function space.

    :arg V: a :class:`FunctionSpace`.
    :arg fields: a list of integers defining subspaces of V.

    :returns: a list of integers with the scalar components corresponding to
    the subfields.
    """
    cur = 0
    components = []
    if len(fields) == 0:
        return components
    for i, Vi in enumerate(V):
        if i in fields:
            components.extend(range(cur, cur+Vi.value_size))
        cur += Vi.value_size
    return components


def replace(e, mapping):
    """A wrapper for ufl.replace that allows numpy arrays and skips
    substitution into sub-expressions wrapped by :func:`~irksome.lag`."""
    cmapping = {k: as_tensor(v) for k, v in mapping.items()}
    for var in extract_type(e, Variable):
        if var.ufl_operands[1] is lag_label:
            cmapping.setdefault(var, var)
    return ufl_replace(e, cmapping)


# Utility functions that help us refactor
def AI(A):
    return (A, numpy.eye(*A.shape, dtype=A.dtype))


def IA(A):
    return (numpy.eye(*A.shape, dtype=A.dtype), A)


def is_ode(f, u):
    """Given a form defined over a function `u`, checks if
    (each bit of) u appears under a time derivative."""
    derivs = extract_type(f, TimeDerivative)
    Dtbits = []
    for k in derivs:
        op, = k.ufl_operands
        Dtbits.extend(op[i] for i in numpy.ndindex(op.ufl_shape))
    ubits = [u[i] for i in numpy.ndindex(u.ufl_shape)]
    return set(ubits) <= set(Dtbits)


def get_lagrange_permutation(L):
    """Given a univariate Lagrange element, return the
    points ordered from left to right and the permutation of the
    dofs required to obtain this re-ordering."""
    assert L.ref_el.get_spatial_dimension() == 1

    points = []
    for ell in L.dual.nodes:
        if not isinstance(ell, FIAT.functional.PointEvaluation):
            raise TypeError("Expecting a Lagrange element")
        pt, = ell.get_point_dict().keys()
        points.append(pt[0])

    c = numpy.asarray(points)
    perm = numpy.argsort(c)

    return c[perm], perm


def get_sub(u: ufl.FunctionSpace | ufl.Coefficient, indices: tuple[int, ...]) -> ufl.FunctionSpace | ufl.Coefficient:
    """Recursively access the subfunction of a mixed function space or the space itself, given the indices of the subspace.

    Args:
        u: A coefficient in a mixed function space
        indices: A tuple of integers giving the indices of the subspace to access. For example
            if u is in a mixed space (V1, V2, V3), then get_sub(u, (0, 2)) will return the third subfunction of u in V1,
            i.e. `u.sub(0).sub(2)`.
    """
    for i in indices:
        if i is not None:
            u = u.sub(i)
    return u
