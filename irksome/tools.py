from operator import mul
from functools import reduce
import numpy
from firedrake import Function, FunctionSpace, VectorSpaceBasis, MixedVectorSpaceBasis, Constant, split
from ufl.algorithms.analysis import extract_type
from ufl.measure import Measure, MeasureSum, integral_type_to_measure_name, measure_name_to_integral_type
from ufl import as_tensor, zero
from ufl import replace as ufl_replace
from pyop2.types import MixedDat

from irksome.deriv import TimeDerivative
from irksome.labeling import TimeQuadratureLabel


def unique_mesh(mesh):
    try:
        mesh, = set(mesh)
    except TypeError:
        pass
    return mesh


def dot(A, B):
    return numpy.tensordot(A, B, (-1, 0))


def reshape(expr, shape):
    return numpy.reshape([expr[i] for i in numpy.ndindex(expr.ufl_shape)], shape)


def flatten_dats(dats):
    flat_dat = []
    for dat in dats:
        if isinstance(dat, (tuple, list, MixedDat)):
            flat_dat.extend(dat)
        else:
            flat_dat.append(dat)
    return MixedDat(flat_dat)


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
        if isinstance(nullspace, (MixedVectorSpaceBasis, VectorSpaceBasis)):
            nullspace = [(field, basis) for field, basis in enumerate(nullspace)
                         if isinstance(basis, VectorSpaceBasis)]
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


def replace_auxiliary_variables(F, u0, aux_indices):
    """Discretize the fields corresponding to aux_indices in Dt(V)."""
    if aux_indices is None:
        return F

    components = []
    for i, usub in enumerate(split(u0)):
        if i in aux_indices:
            usub = TimeDerivative(usub)
        components.extend(usub[i] for i in numpy.ndindex(usub.ufl_shape))
    return replace(F, {u0: numpy.reshape(components, u0.ufl_shape)})


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
        self.msh = unique_mesh(msh)
        self.V = FunctionSpace(self.msh, 'R', 0)

    def Constant(self, val=0.0):
        return Function(self.V).assign(val)


def ConstantOrZero(x, MC=None):
    const = MC.Constant if MC else Constant
    return zero() if abs(complex(x)) < 1.e-10 else const(x)


vecconst = numpy.vectorize(ConstantOrZero)


# Measure for space-time integration with Galerkin-in-time
class TimeMeasure(Measure):
    """Representation of a space-time integration measure.
    """

    # Only declare new attributes introduced in this subclass.
    # Parent Measure already defines slots for domain/integral/metadata/etc.
    __slots__ = ("time_degree", "quad_rule", "space_measure")

    def __init__(
        self,
        integral_type,  # "dx" etc
        time_degree,
        domain=None,
        subdomain_id="everywhere",
        metadata=None,
        subdomain_data=None,
    ):
        """Initialise.

        Args:
            integral_type: one of "cell", etc, or short form "dx", etc
            time_degree: degree of time quadrature
            domain: an AbstractDomain object (most often a Mesh)
            subdomain_id: either string "everywhere", a single subdomain id int, or tuple of ints
            metadata: dict, with additional compiler-specific parameters
                affecting how code is generated, including parameters
                for optimization or debugging of generated code
            subdomain_data: object representing data to interpret subdomain_id with
        """
        self.time_degree = time_degree
        self.quad_rule = TimeQuadratureLabel(time_degree)
        self.space_measure = Measure(
            integral_type,
            domain=domain,
            subdomain_id=subdomain_id,
            metadata=metadata,
            subdomain_data=subdomain_data
        )
        super().__init__(
            integral_type,
            domain=domain,
            subdomain_id=subdomain_id,
            metadata=metadata,
            subdomain_data=subdomain_data
        )

    def __add__(self, other):
        """Add two measures (self+other).

        Creates an intermediate object used for the notation
          expr * (dx(1) + dx(2)) := expr * dx(1) + expr * dx(2)
        """
        if isinstance(other, TimeMeasure):
            # Let dx(1) + dx(2) equal dx((1,2))
            return TimeMeasureSum(self, other)
        else:
            # Can only add Measures
            return NotImplemented

    def __mul__(self, other):
        """Multiply two space-time measures (self*other).

        Not yet functional.
        """
        return NotImplemented

    def __rmul__(self, integrand):
        """Multiply a scalar expression with measure to construct a form with a single integral.
        """
        return self.quad_rule(integrand * self.space_measure)

class TimeMeasureSum(MeasureSum):
    """Represents a sum of space-time measures.
    """

    def __init__(self, *measures):
        """Initialise."""
        super().__init__(*measures)

    def __add__(self, other):
        """Add."""
        if isinstance(other, TimeMeasure):
            return TimeMeasureSum(*(self._measures + (other,)))
        elif isinstance(other, TimeMeasureSum):
            return TimeMeasureSum(*(self._measures + other._measures))
        return NotImplemented

def _make_time_measure_function(name, integral_type):
    """Convert integral type into function
    """
    # e.g. name="dx", integral_type="cell"
    def _f(time_degree, domain=None, subdomain_id="everywhere",
           metadata=None, subdomain_data=None):
        return TimeMeasure(
            integral_type=integral_type,
            time_degree=time_degree,
            domain=domain,
            subdomain_id=subdomain_id,
            metadata=metadata,
            subdomain_data=subdomain_data
        )
    _f.__name__ = f"{name}_time"
    return _f

for _name, _itype in measure_name_to_integral_type.items():
    """Export all time-measure functions into the module namespace
    """
    globals()[f"{_name}_time"] = _make_time_measure_function(_name, _itype)
