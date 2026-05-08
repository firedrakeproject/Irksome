"""Firedrake backend for Irksome"""

from operator import mul
from functools import reduce

import firedrake
import ufl
from ..tools import get_stage_space
import typing


def get_stage_space(V: ufl.FunctionSpace, num_stages: int) -> ufl.FunctionSpace:
    return reduce(mul, (V for _ in range(num_stages)))


def extract_bcs(bcs: typing.Any) -> tuple[typing.Any]:
    """Return an iterable of boundary conditions on the residual form"""
    return tuple(bc.extract_form("F") for bc in firedrake.solving._extract_bcs(bcs))


def create_nonlinearvariational_problem(
    F: ufl.Form,
    u: ufl.Coefficient,
    bcs: typing.Sequence | None = None,
    **kwargs,
) -> firedrake.NonlinearVariationalProblem:
    return firedrake.NonlinearVariationalProblem(F, u, bcs=bcs, **kwargs)


def create_nonlinearvariational_solver(
    problem: firedrake.NonlinearVariationalProblem,
    solver_parameters: dict | None = None,
    **kwargs,
):
    """Create a non-linear variational solver that uses PETSc SNES."""
    return firedrake.NonlinearVariationalSolver(
        problem, solver_parameters=solver_parameters, **kwargs
    )


def create_linearvariational_problem(
    a: ufl.Form,
    L: ufl.Form,
    u: ufl.Coefficient | typing.Sequence[ufl.Coefficient],
    bcs: typing.Sequence | None = None,
    aP: ufl.Form | None = None,
    **kwargs,
) -> firedrake.LinearVariationalProblem:
    return firedrake.LinearVariationalProblem(a, L, u, bcs=bcs, aP=aP, **kwargs)


def create_linearvariational_solver(
    problem: firedrake.LinearVariationalProblem,
    solver_parameters: dict | None = None,
    **kwargs,
):
    """Create a linear variational solver that uses PETSc KSP."""
    return firedrake.LinearVariationalSolver(
        problem, solver_parameters=solver_parameters, **kwargs
    )


def get_function_space(u: ufl.Coefficient) -> firedrake.FunctionSpace:
    return u.function_space()


def get_stages(V: firedrake.FunctionSpace, num_stages: int) -> firedrake.Function:
    """
    Given a function space for a single time-step, get a duplicate of this space,
    repeated `num_stages` times.

    Args:
        V: Space for single step
        num_stages: Number of stages

    Returns:
        A coefficient in the new function space
    """
    Vbig = get_stage_space(V, num_stages)
    return firedrake.Function(Vbig)


class MeshConstant(object):
    def __init__(self, msh: ufl.Mesh):
        self.msh = ufl.domain.as_domain(msh)
        self.V = firedrake.FunctionSpace(self.msh, "R", 0)

    def Constant(self, val=0.0) -> ufl.Coefficient:
        return firedrake.Function(self.V).assign(val)


def get_mesh_constant(MC: MeshConstant | None):
    return MC.Constant if MC else firedrake.Constant


def invalidate_jacobian(solver: firedrake.LinearVariationalSolver):
    """Invalidate the Jacobian matrix in the backend language."""
    firedrake.LinearVariationalSolver.invalidate_jacobian(solver)


Function = firedrake.Function

DirichletBC = firedrake.DirichletBC

norm = firedrake.norm

assemble = firedrake.assemble

replace = firedrake.replace

derivative = firedrake.derivative
TestFunction = firedrake.TestFunction
TrialFunction = firedrake.TrialFunction
