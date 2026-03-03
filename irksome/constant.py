import numpy as np
from .backend import get_backend, Backend
import ufl


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
    return ufl.zero() if abs(complex(x)) < 1.0e-10 else const(x)


vecconst = np.vectorize(ConstantOrZero)
