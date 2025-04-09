from operator import mul
from functools import reduce
import numpy
from firedrake import Function, FunctionSpace, MixedVectorSpaceBasis, Constant
from ufl.algorithms.analysis import extract_type
from ufl import as_tensor, zero
from ufl import replace as ufl_replace
from pyop2.types import MixedDat

from irksome.deriv import TimeDerivative


def flatten_dats(dats):
    flat_dat = []
    for dat in dats:
        if isinstance(dat, MixedDat):
            flat_dat.extend(dat.split)
        elif isinstance(dat, (tuple, list)):
            flat_dat.extend(dat)
        else:
            flat_dat.append(dat)
    return flat_dat


def get_stage_space(V, num_stages):
    return reduce(mul, (V for _ in range(num_stages)))


def getNullspace(V, Vbig, num_stages, nullspace):
    """
    Computes the nullspace for a multi-stage method.

    :arg V: The :class:`FunctionSpace` on which the original time-dependent PDE is posed.
    :arg Vbig: The multi-stage :class:`FunctionSpace` for the stage problem
    :arg num_stages: The number of stages in the RK method
    :arg nullspace: The nullspace for the original problem.

    On output, we produce a :class:`MixedVectorSpaceBasis` defining the nullspace
    for the multistage problem.
    """

    num_fields = len(V)
    if nullspace is None:
        nspnew = None
    else:
        try:
            nullspace.sort()
        except AttributeError:
            raise AttributeError("Nullspace entries must be of form (idx, VSP), where idx is a non-negative integer")
        if (nullspace[-1][0] > num_fields) or (nullspace[0][0] < 0):
            raise ValueError("At least one index for nullspaces is out of range")
        nspnew = []
        nsp_comp = len(nullspace)
        for i in range(num_stages):
            count = 0
            for j in range(num_fields):
                if count < nsp_comp and j == nullspace[count][0]:
                    nspnew.append(nullspace[count][1])
                    count += 1
                else:
                    nspnew.append(Vbig.sub(j + num_fields * i))
        nspnew = MixedVectorSpaceBasis(Vbig, nspnew)

    return nspnew


def replace(e, mapping):
    """A wrapper for ufl.replace that allows numpy arrays."""
    cmapping = {k: as_tensor(v) for k, v in mapping.items()}
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
    return set(Dtbits) == set(ubits)


# Utility class for constants on a mesh
class MeshConstant(object):
    def __init__(self, msh):
        self.msh = msh
        self.V = FunctionSpace(msh, 'R', 0)

    def Constant(self, val=0.0):
        return Function(self.V).assign(val)


def ConstantOrZero(x, MC=None):
    const = MC.Constant if MC else Constant
    return zero() if abs(complex(x)) < 1.e-10 else const(x)


vecconst = numpy.vectorize(ConstantOrZero)
