from typing import Protocol, Any, Sequence
import ufl
from importlib import import_module


class Backend(Protocol):
    def get_function_space(self, V: ufl.Coefficient) -> ufl.FunctionSpace:
        """Get a function space from the backend"""

    def extract_bcs(bcs: Any)->tuple[Any]:
        """Extract boundary conditions"""
 
    class Function:
        ...

    class DirichletBC:
        ...
    def get_stages(self, V: ufl.FunctionSpace, num_stages: int) -> ufl.Coefficient:
        """
        Given a function space for a single time-step, get a duplicate of this space,
        repeated `num_stages` times.

        Args:
            V: Space for single step
            num_stages: Number of stages

        Returns:
            A coefficient in the new function space
        """

    class MeshConstant:
        def __init__(self, msh: ufl.Mesh):
            """Initialize a mesh constant over a domain"""

        def Constant(self, val: float = 0.0):
            """Generate a constant in the backend language with a specific value"""

    def ConstantOrZero(
        x: float | complex, MC: MeshConstant | None = None
    ) -> ufl.core.expr.Expr:
        """
        Create a constant with backend class if MeshConstant is not supplied.
        Create UFL zero if `x` is sufficiently small
        """

    def get_mesh_constant(MC: MeshConstant | None) -> ufl.core.expr.Expr:
        """Get a backend class to construct a mesh constant from"""

    def TestFunction(space: ufl.FunctionSpace, part: int|None=None)-> ufl.Argument:
        """Return a test-function that can be used by forms in the backend."""

    def create_nonlinearvariational_solver(F: ufl.Form, u: ufl.Coefficient, bcs: DirichletBC | Sequence | None = None, solver_parameters: dict | None = None, **kwargs):
        """Create a non-linear variational solver that uses PETSc SNES."""

    def get_stage_spaces(V: ufl.FunctionSpace, num_stages: int) -> ufl.FunctionSpace:
        """Create a stage space with M number of components."""

def get_backend(backend: str) -> Backend:
    """Get backend class from backend name.

    Args:
        backend: Name of the backend to get

    Returns:
        Backend class
    """
    if backend == "firedrake":
        from .backends import firedrake as fd_backend

        return fd_backend
    elif backend == "dolfinx":
        from .backends import dolfinx as dx_backend

        return dx_backend
    else:
        return import_module(backend)
