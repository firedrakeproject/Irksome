from typing import Protocol, Any, Sequence
import ufl
from importlib import import_module
import types


class Backend(Protocol):
    def get_function_space(self, V: ufl.Coefficient) -> ufl.FunctionSpace:
        """Get a function space from the backend"""

    def extract_bcs(bcs: Any) -> tuple[Any]:
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

    class Constant:
        """MeshLess constant class"""

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

    def TestFunction(space: ufl.FunctionSpace, part: int | None = None) -> ufl.Argument:
        """Return a test-function that can be used by forms in the backend."""

    def TrialFunction(
        space: ufl.FunctionSpace, part: int | None = None
    ) -> ufl.Argument:
        """Return a trial-function that can be used by forms in the backend."""

    def create_variational_problem(
        F: ufl.Form,
        u: ufl.Coefficient,
        bcs: DirichletBC | Sequence | None = None,
        **kwargs,
    ) -> Any:
        """Create a variational problem in the backend language."""

    def create_variational_solver(
        problem: Any,
        solver_parameters: dict | None = None,
        **kwargs,
    ):
        """Create a variational solver in the backend language."""

    def get_stage_spaces(V: ufl.FunctionSpace, num_stages: int) -> ufl.FunctionSpace:
        """Create a stage space with M number of components."""

    def norm(
        v: ufl.core.expr.Expr, norm_type: str = "L2", mesh: ufl.Mesh | None = None
    ) -> float:
        """Compute the norm of a function in the backend language."""

    def assemble(expr: ufl.core.expr.Expr) -> Any:
        """Assemble a UFL expression in the backend language."""

    def derivative(
        form: ufl.Form,
        u: ufl.Coefficient,
        du: ufl.Argument | None = None,
        coefficient_derivatives: dict | None = None,
    ) -> ufl.Form:
        """Compute the derivative of a form with respect to a coefficient in the backend language."""

    def invalidate_jacobian(solver: Any):
        """Invalidate the Jacobian matrix in the backend language."""

    class EquationBCSplit:
        ...

    class EquationBC:
        ...


def get_backend(backend: str | types.ModuleType) -> Backend:
    """Get backend class from backend name.

    Args:
        backend: Name of the backend to get

    Returns:
        Backend class
    """
    if isinstance(backend, types.ModuleType):
        return backend
    if backend == "firedrake":
        from .backends import firedrake as fd_backend

        return fd_backend
    elif backend == "dolfinx":
        from .backends import dolfinx as dx_backend

        return dx_backend
    else:
        return import_module(backend)
