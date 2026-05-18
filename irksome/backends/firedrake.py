"""Firedrake backend for Irksome"""

import firedrake
import ufl
import typing
from functools import reduce
from operator import mul


assemble = firedrake.assemble
derivative = firedrake.derivative
norm = firedrake.norm
Function = firedrake.Function
TestFunction = firedrake.TestFunction
TrialFunction = firedrake.TrialFunction
DirichletBC = firedrake.DirichletBC
EquationBC = firedrake.bcs.EquationBC
EquationBCSplit = firedrake.bcs.EquationBCSplit
Constant = firedrake.Constant


def get_stage_space(V: ufl.FunctionSpace, num_stages: int) -> ufl.FunctionSpace:
    # NOTE: Should use firedrake.MixedFunctionSpace, but there is a bug for vector spaces.
    return reduce(mul, (V for _ in range(num_stages)))


def extract_bcs(bcs: typing.Any) -> tuple[typing.Any]:
    """Return an iterable of boundary conditions on the residual form"""
    return tuple(bc.extract_form("F") for bc in firedrake.solving._extract_bcs(bcs))


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


def create_variational_problem(F, u, bcs=None, J=None, Jp=None, **kwargs):
    if len(F.arguments()) == 2:
        a, L = ufl.system(F)
        kwargs.pop("is_linear", None)
        problem = firedrake.LinearVariationalProblem(a, L, u, bcs=bcs, aP=Jp, **kwargs)
    else:
        constant_jacobian = kwargs.pop("constant_jacobian", False)
        problem = firedrake.NonlinearVariationalProblem(F, u, bcs=bcs, J=J, Jp=Jp, **kwargs)
        if constant_jacobian:
            problem._constant_jacobian = constant_jacobian
    return problem


def create_variational_solver(problem, **kwargs):
    if isinstance(problem, firedrake.LinearVariationalProblem):
        return firedrake.LinearVariationalSolver(problem, **kwargs)
    else:
        return firedrake.NonlinearVariationalSolver(problem, **kwargs)


def invalidate_jacobian(solver):
    return firedrake.LinearVariationalSolver.invalidate_jacobian(solver)


class BoundsConstrainedDirichletBC(firedrake.DirichletBC):
    def __init__(
        self,
        V,
        g,
        sub_domain,
        bounds,
        solver_parameters=None,
    ):
        self.g = g
        self.solver_parameters = solver_parameters
        self.bounds = bounds
        self.gnew = Function(V)

        F = ufl.inner(self.gnew - g, TestFunction(V)) * ufl.dx

        if solver_parameters is None:
            solver_parameters = {
                "snes_type": "vinewtonrsls",
                "snes_max_it": 300,
                "snes_atol": 1.0e-8,
                "ksp_type": "preonly",
                "mat_type": "aij",
            }
        problem = create_variational_problem(F, self.gnew)
        self.solver = create_variational_solver(
            problem, solver_parameters=solver_parameters
        )
        super().__init__(V, g, sub_domain)

    @property
    def function_arg(self):
        """The value of this boundary condition."""
        self.solver.solve(bounds=self.bounds)
        return self.gnew

    @function_arg.setter
    def function_arg(self, g):
        """Set the value of this boundary condition."""
        self.solver.solve(bounds=self.bounds)
        return self.gnew

    def reconstruct(self, V=None, g=None, sub_domain=None):
        V = V or self.function_space()
        g = g or self.g
        sub_domain = sub_domain or self.sub_domain
        return type(self)(V, g, sub_domain, self.bounds, self.solver_parameters)


def create_bounds_constrained_bc(V, g, sub_domain, bounds, solver_parameters=None):
    return BoundsConstrainedDirichletBC(V, g, sub_domain, bounds, solver_parameters=solver_parameters)
