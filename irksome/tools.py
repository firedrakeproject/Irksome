import numpy
from firedrake import Function, FunctionSpace, MixedVectorSpaceBasis, split
from ufl.algorithms.analysis import extract_type, has_exact_type
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import CoefficientDerivative
from ufl.constantvalue import as_ufl
from ufl.corealg.multifunction import MultiFunction
from irksome.deriv import TimeDerivative


def getNullspace(V, Vbig, butch, nullspace):
    """
    Computes the nullspace for a multi-stage method.

    :arg V: The :class:`FunctionSpace` on which the original time-dependent PDE is posed.
    :arg Vbig: The multi-stage :class:`FunctionSpace` for the stage problem
    :arg butch: The :class:`ButcherTableau` defining the RK method
    :arg nullspace: The nullspace for the original problem.

    On output, we produce a :class:`MixedVectorSpaceBasis` defining the nullspace
    for the multistage problem.
    """

    num_stages = butch.num_stages
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
        for i in range(num_stages):
            count = 0
            for j in range(num_fields):
                if j == nullspace[count][0]:
                    nspnew.append(nullspace[count][1])
                    count += 1
                else:
                    nspnew.append(Vbig.sub(j + num_fields * i))
        nspnew = MixedVectorSpaceBasis(Vbig, nspnew)

    return nspnew


# Update for UFL's replace that performs post-order traversal and hence replaces
# more complicated expressions first.
class MyReplacer(MultiFunction):
    def __init__(self, mapping):
        super().__init__()
        self.replacements = mapping
        if not all(k.ufl_shape == v.ufl_shape for k, v in mapping.items()):
            raise ValueError("Replacement expressions must have the same shape as what they replace.")

    def expr(self, o):
        if o in self.replacements:
            return self.replacements[o]
        else:
            return self.reuse_if_untouched(o, *map(self, o.ufl_operands))


def replace(e, mapping):
    """Replace subexpressions in expression.

    @param e:
        An Expr or Form.
    @param mapping:
        A dict with from:to replacements to perform.
    """
    mapping2 = dict((k, as_ufl(v)) for (k, v) in mapping.items())

    # Workaround for problem with delayed derivative evaluation
    # The problem is that J = derivative(f(g, h), g) does not evaluate immediately
    # So if we subsequently do replace(J, {g: h}) we end up with an expression:
    # derivative(f(h, h), h)
    # rather than what were were probably thinking of:
    # replace(derivative(f(g, h), g), {g: h})
    #
    # To fix this would require one to expand derivatives early (which
    # is not attractive), or make replace lazy too.
    if has_exact_type(e, CoefficientDerivative):
        # Hack to avoid circular dependencies
        from ufl.algorithms.ad import expand_derivatives
        e = expand_derivatives(e)

    return map_integrand_dags(MyReplacer(mapping2), e)


# Utility functions that help us refactor
def AI(A):
    return (A, numpy.eye(*A.shape, dtype=A.dtype))


def IA(A):
    return (numpy.eye(*A.shape, dtype=A.dtype), A)


def is_ode(f, u):
    """Given a form defined over a function `u`, checks if
    (each bit of) u appears under a time derivative."""
    blah = extract_type(f, TimeDerivative)

    Dtbits = set(b.ufl_operands[0] for b in blah)
    ubits = set(split(u))
    return Dtbits == ubits


# Utility class for constants on a mesh
class MeshConstant(object):
    def __init__(self, msh):
        self.msh = msh
        self.V = FunctionSpace(msh, 'R', 0)

    def Constant(self, val=0.0):
        return Function(self.V).assign(val)


# used to figure out how to apply Dirichlet BC to each stage
def stage2spaces4bc(bc, V, Vbig, i):
    num_fields = len(V)
    if num_fields == 1:  # not mixed space
        comp = bc.function_space().component
        if comp is not None:  # check for sub-piece of vector-valued
            Vsp = V.sub(comp)
            Vbigi = Vbig[i].sub(comp)
        else:
            Vsp = V
            Vbigi = Vbig[i]
    else:  # mixed space
        sub = bc.function_space_index()
        comp = bc.function_space().component
        if comp is not None:  # check for sub-piece of vector-valued
            Vsp = V.sub(sub).sub(comp)
            Vbigi = Vbig[sub+num_fields*i].sub(comp)
        else:
            Vsp = V.sub(sub)
            Vbigi = Vbig[sub+num_fields*i]

    return Vsp, Vbigi
