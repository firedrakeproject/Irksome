"""DOLFINx backend for Irksome"""

try:
    from mpi4py import MPI
    import basix.ufl
    import dolfinx.fem.petsc
    import ufl
    import typing
    import numpy as np

    def get_stage_space(V: ufl.FunctionSpace, num_stages: int) -> ufl.FunctionSpace:
        if num_stages == 1:
            me = V.ufl_elemet()
        else:
            el = V.ufl_element()
            if el.num_sub_elements > 0:
                me = basix.ufl.mixed_element(
                    np.tile(el.sub_elements, num_stages).tolist()
                )
            else:
                me = basix.ufl.blocked_element(el, shape=(num_stages,))
        return dolfinx.fem.functionspace(V.mesh, me)

    def extract_bcs(bcs: typing.Any) -> tuple[typing.Any]:
        """Extract boundary conditions"""
        return bcs

    def create_linearvariational_problem(
        a: ufl.Form,
        L: ufl.Form,
        u: ufl.Coefficient | typing.Sequence[ufl.Coefficient],
        bcs: typing.Sequence | None = None,
        aP: ufl.Form | None = None,
        **kwargs,
    ) -> dolfinx.fem.petsc.LinearProblem:
        return dolfinx.fem.petsc.LinearProblem(
            a,
            L,
            u,
            bcs=bcs,
            petsc_options_prefix="IrkSomeLinearSolver",
            P=aP,
            **kwargs,
        )

    def create_linear_solver(
        problem: dolfinx.fem.petsc.LinearProblem,
        solver_parameters: dict | None = None,
        **kwargs,
    ):
        """Create a linear variational solver that uses PETSc KSP."""
        return problem

    def create_nonlinearvariational_problem(
        F: ufl.Form,
        g: ufl.Coefficient,
        bcs: typing.Sequence | None = None,
        solver_parameters: dict | None = None,
    ) -> dolfinx.fem.petsc.NonlinearProblem:
        return dolfinx.fem.petsc.NonlinearProblem(
            F,
            g,
            petsc_options_prefix="IrkSomeNonlinearSolver",
            bcs=bcs,
            petsc_options=solver_parameters,
        )

    def create_nonlinear_solver(
        problem: dolfinx.fem.petsc.NonlinearProblem,
        solver_parameters: dict | None = None,
    ):
        """Create a non-linear variational solver that uses PETSc SNES."""
        return problem

    def get_function_space(u: ufl.Coefficient) -> ufl.FunctionSpace:
        return u.ufl_function_space()

    def get_stages(V: dolfinx.fem.FunctionSpace, num_stages: int) -> ufl.Coefficient:
        """
        Given a function space for a single time-step, get a duplicate of this space,
        repeated `num_stages` times.

        Args:
            V: Space for single step
            num_stages: Number of stages

        Returns:
            A coefficient in the new function space
        """
        if V.num_sub_spaces == 0:
            el = basix.ufl.mixed_element([V.ufl_element()] * num_stages)
        else:
            el = basix.ufl.mixed_element(V.ufl_element().sub_elements * num_stages)
        Vbig = dolfinx.fem.functionspace(V.mesh, el)
        return dolfinx.fem.Function(Vbig)

    class MeshConstant(object):
        def __init__(self, msh):
            self.msh = msh
            try:
                import basix.ufl

                r_el = basix.ufl.real_element(
                    msh.basix_cell(), value_shape=(), dtype=dolfinx.default_scalar_type
                )
                self.V = dolfinx.fem.functionspace(msh, r_el)
            except TypeError:
                try:
                    import scifem
                except ModuleNotFoundError:
                    raise RuntimeError(
                        "DOLFINx with real element support or Scifem is required to make mesh-constants"
                    )
                self.V = scifem.create_real_functionspace(msh, ())

        def Constant(self, val=0.0) -> ufl.Coefficient:
            v = dolfinx.fem.Function(self.V)
            v.value = val
            return v

    def get_mesh_constant(MC: MeshConstant | None) -> ufl.core.expr.Expr:
        return MC.Constant if MC is not None else ufl.constantvalue.ComplexValue

    class DirichletBC(dolfinx.fem.DirichletBC):
        pass

    def norm(
        v: ufl.core.Expr, norm_type: str = "L2", mesh: ufl.Mesh | None = None
    ) -> float:
        """Compute the norm of a function in the backend language."""
        if mesh is not None:
            dx = ufl.Mesure("dx", domain=mesh)
        else:
            dx = ufl.dx
        p = 2
        if norm_type.startswith("L"):
            p = int(norm_type[1:])
            if p < 1:
                raise ValueError(f"Invalid norm type {norm_type}")
            expr = ufl.inner(v, v) ** (p / 2)
            form = dolfinx.fem.form(expr * dx)
        else:
            raise NotImplementedError(f"Norm type {norm_type} not implemented")
        norm_loc = dolfinx.fem.assemble_scalar(form)
        return form.mesh.comm.Allreduce(MPI.IN_PLACE, norm_loc, op=MPI.SUM) ** (1 / p)

    def assemble(expr: ufl.core.Expr | float):
        """Assemble a UFL expression in the backend language."""
        if isinstance(expr, float):
            return float
        else:
            form = dolfinx.fem.form(expr)
            if form.rank == 0:
                return dolfinx.fem.assemble_scalar(form)
            elif form.rank == 1:
                return dolfinx.fem.assemble_vector(form)
            elif form.rank == 2:
                return dolfinx.fem.assemble_matrix(form)
            else:
                raise ValueError(f"Cannot assemble form of rank {form.rank}")

    derivative = ufl.derivative
    TrialFunction = ufl.TrialFunction
    Function = dolfinx.fem.Function
    TestFunction = ufl.TestFunction

    def invalidate_jacobian(solver: dolfinx.fem.petsc.LinearProblem):
        """Invalidate the Jacobian matrix in the backend language."""
        raise RuntimeError("DOLFINx does not support Jacobian invalidation")
except ModuleNotFoundError:
    pass
