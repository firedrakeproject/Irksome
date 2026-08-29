import numpy as np
from .backend import get_backend, Backend
import ufl


# A tableau coefficient smaller than this is dropped from the generated form.
STRUCTURAL_ZERO = 1.0e-10


def MeshConstant(msh, backend: str = "firedrake"):
    mc_backend = get_backend(backend)
    return mc_backend.MeshConstant(msh)


def ConstantOrZero(
    x: float | complex,
    MC: Backend.MeshConstant | None = None,
    backend: str = "firedrake",
) -> ufl.core.expr.Expr:
    backend_impl = get_backend(backend)
    const = backend_impl.get_mesh_constant(MC)
    return ufl.zero() if abs(complex(x)) < STRUCTURAL_ZERO else const(x)


vecconst = np.vectorize(ConstantOrZero)
