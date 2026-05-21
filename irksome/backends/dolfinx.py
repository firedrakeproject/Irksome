"""DOLFINx backend for Irksome"""

from collections.abc import Sequence

from irksome.tools import get_sub

try:
    from mpi4py import MPI
    from petsc4py import PETSc
    import dolfinx.fem.petsc
    from dolfinx.typing import Scalar

    import ufl
    import typing
    import numpy as np
    import numpy.typing as npt

    # Patching of DOLFINx objects to mimick firedrake naming and properties.
    def function_space_length(self):
        num_sub_elements = self.ufl_element().num_sub_elements
        return 1 if num_sub_elements == 0 else num_sub_elements

    def mixed_space_length(self):
        return len(self.ufl_sub_spaces())

    def subfunctions(self):
        """Get subfunctions for a DOLFINx function, which may be in a mixed space."""
        if self.function_space.ufl_element().num_sub_elements == 0:
            return [self]
        else:
            return [self.sub(i) for i in range(self.function_space().ufl_element().num_sub_elements)]

    dolfinx.fem.FunctionSpace.__len__ = function_space_length
    dolfinx.fem.Function.subfunctions = property(subfunctions)
    ufl.MixedFunctionSpace.__len__ = mixed_space_length

    class ListTensor(ufl.tensors.ListTensor):
        """A list tensor that exposes subfunctions for DOLFINx functions"""
        @property
        def subfunctions(self):
            return [self.ufl_operands[i] for i in range(len(self))]

        def function_space(self):
            return ufl.MixedFunctionSpace(*[self.ufl_operands[i].ufl_function_space() for i in range(len(self))])

    class LinearProblem(dolfinx.fem.petsc.LinearProblem):

        def solve(self, bounds=None):
            if bounds is not None:
                raise NotImplementedError("Bounds-constrained solves are not implemented for DOLFINx")
            [bc.pack() for bc in self._bcs]
            super().solve()

    class NonlinearProblem(dolfinx.fem.petsc.NonlinearProblem):

        def __init__(
            self,
            F: ufl.form.Form | Sequence[ufl.form.Form],
            u: dolfinx.fem.Function | Sequence[dolfinx.fem.Function],
            *,
            petsc_options_prefix: str,
            bcs: Sequence[dolfinx.fem.DirichletBC] | None = None,
            J: ufl.form.Form | Sequence[Sequence[ufl.form.Form]] | None = None,
            P: ufl.form.Form | Sequence[Sequence[ufl.form.Form]] | None = None,
            kind: str | Sequence[Sequence[str]] | None = None,
            petsc_options: dict | None = None,
            form_compiler_options: dict | None = None,
            jit_options: dict | None = None,
            entity_maps: Sequence[dolfinx.mesh.EntityMap] | None = None,
        ):
            super().__init__(
                F,
                u,
                petsc_options_prefix=petsc_options_prefix,
                bcs=bcs,
                J=J,
                P=P,
                kind=kind,
                petsc_options=petsc_options,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            )
            _vec, (current_func, args, kargs) = self.solver.getFunction()

            def assemble_residual(
                    _snes: PETSc.SNES,  # type: ignore[name-defined]
                    x: PETSc.Vec,  # type: ignore[name-defined]
                    b: PETSc.Vec,  # type: ignore[name-defined]
                    u: dolfinx.fem.Function | Sequence[dolfinx.fem.Function],
                    residual: dolfinx.fem.Form | Sequence[dolfinx.fem.Form],
                    jacobian: dolfinx.fem.Form | Sequence[Sequence[dolfinx.fem.Form]],
                    bcs: Sequence[dolfinx.fem.DirichletBC],
                    _blocks: tuple[tuple[int, int, int], ...] | None = None,):
                [bc.pack() for bc in bcs]
                current_func(_snes, x, b, u, residual, jacobian, bcs, _blocks)
            self.solver.setFunction(assemble_residual, _vec, args=args, kargs=kargs)

            _jac, _jacP, (current_jac, args, kargs) = self.solver.getJacobian()

            def assemble_jacobian(
                    _snes: PETSc.SNES,  # type: ignore[name-defined]
                    x: PETSc.Vec,  # type: ignore[name-defined]
                    J: PETSc.Mat,  # type: ignore[name-defined]
                    P_mat: PETSc.Mat,  # type: ignore[name-defined]
                    u: Sequence[dolfinx.fem.Function] | dolfinx.fem.Function,
                    jacobian: dolfinx.fem.Form | Sequence[Sequence[dolfinx.fem.Form]],
                    preconditioner: dolfinx.fem.Form | Sequence[Sequence[dolfinx.fem.Form]] | None,
                    bcs: Sequence[dolfinx.fem.DirichletBC],
            ):
                [bc.pack() for bc in bcs]
                current_jac(_snes, x, J, P_mat, u, jacobian, preconditioner, bcs)
            self.solver.setJacobian(assemble_jacobian, _jac, _jacP, args=args, kargs=kargs)

        def solve(self, bounds=None):
            if bounds is not None:
                raise NotImplementedError("Bounds-constrained solves are not implemented for DOLFINx")
            super().solve()

        @property
        def snes(self) -> PETSc.SNES:  # type: ignore[name-defined]
            return self.solver

    def dirichletbc(
        value: dolfinx.fem.Function
        | dolfinx.fem.Constant
        | npt.NDArray[Scalar]
        | float
        | complex,
        dofs: npt.NDArray[np.int32],
        V: dolfinx.fem.FunctionSpace | None = None,
    ):
        """Overloaded DirichletBC so that we can reconstruct BCs with UFL expressions"""
        bc = dolfinx.fem.dirichletbc(value, dofs, V)
        bc._ufl_space = V if V is not None else value.ufl_function_space()
        return bc

    def get_stage_space(V: ufl.FunctionSpace, num_stages: int) -> ufl.FunctionSpace:
        if num_stages == 1:
            space_list = [V]
        else:
            space_list = [V.clone() for _ in range(num_stages)]
        return ufl.MixedFunctionSpace(*space_list)

    class DirichletBC(dolfinx.fem.DirichletBC):
        _pack_expression: dolfinx.fem.Expression | None

        def __init__(self, bc: dolfinx.fem.DirichletBC, V=None, new_value=None):
            """
            Create an Irksome compatible DirichletBC from an existing DOLFINx bc, created by `irksome.backends.dolfinx.dirichletbc`.

            Args:
                bc: A DOLFINx DirichletBC object
                V: A function space V to reconstruct the BC on.
                new_value: A new value to reconstruct the BC with.

            """

            # Attach UFL function space (to be able to reconstruct functions and constants on the same UFL domain)
            if not hasattr(bc, "_ufl_space"):
                if V is not None:
                    bc._ufl_space = V
                else:
                    raise RuntimeError(
                        "Dirichlet condition must be constructed with `irksome.backends.dolfinx.dirichletbc` in order to be reconstructable with UFL expressions"
                    )
            self._ufl_space = bc._ufl_space

            # Get dof indices of existing BC
            dof_indices = bc.dof_indices()[0].copy()

            # If reconstructing use V as the new space rather than the BC space
            bc_space = bc.function_space if V is None else V

            # If we are not reconstructing the BC with a new value, we can reuse existing C++ objects
            if new_value is None:
                val = bc.g
                self._pack_expression = None
            else:
                # If we are reconstructing the BC with a new value, we need to check if the new value is a DOLFINx function or Constant.
                # If True, we do not need to do anything for reconstruction.
                if isinstance(new_value, (dolfinx.fem.Function, dolfinx.fem.Constant)):
                    val = new_value
                    self._pack_expression = None
                else:
                    # If not, we need to take the ufl.core.expr.Expr and pack it into a DOLFINx Expression
                    if bc_space.component() != []:
                        # If working with a subspace of a single stage, we need to create the (parent_dof, sub_dof) mapping
                        V_sub, sub_to_parent = bc_space.collapse()
                        if len(sub_to_parent) != 1:
                            raise NotImplementedError(
                                "Mixed topology is not supported for reconstructing BCs with UFL expressions"
                            )
                        else:
                            sub_to_parent = sub_to_parent[0]
                            parent_to_sub = np.full(
                                bc_space.dofmap.index_map.size_local
                                * bc_space.dofmap.index_map_bs,
                                -1,
                                dtype=np.int32,
                            )
                            parent_to_sub[sub_to_parent] = np.arange(len(sub_to_parent))

                            dof_indices = (dof_indices, parent_to_sub[dof_indices])
                            val = dolfinx.fem.Function(V_sub, name=f"bc_{str(new_value)}")._cpp_object
                    else:
                        val = dolfinx.fem.Function(bc_space, name=f"bc_{str(new_value)}")._cpp_object
                    self._pack_expression = dolfinx.fem.Expression(
                        new_value, bc.function_space.element.interpolation_points()
                    )

            # Reconstruct the C++ object
            # Note: We compare against bc._cpp_object.function_space, NOT the class type.
            cpp_class = type(bc._cpp_object)
            if (
                isinstance(
                    val,
                    (
                        dolfinx.cpp.fem.Function_complex128,
                        dolfinx.cpp.fem.Function_complex64,
                        dolfinx.cpp.fem.Function_float32,
                        dolfinx.cpp.fem.Function_float64,
                    ),
                )
                and val.function_space == bc_space._cpp_object
            ):
                new_cpp_object = cpp_class(val, dof_indices)
            else:
                # Depending on your FEniCSx version, the C++ constructor might strictly
                # expect the C++ FunctionSpace instead of the Python FunctionSpace wrapper.
                try:
                    new_cpp_object = cpp_class(val, dof_indices, bc_space)
                except TypeError:
                    new_cpp_object = cpp_class(val, dof_indices, bc_space._cpp_object)

            # 4. Initialize the parent dolfinx.fem.DirichletBC wrapper with the newly minted C++ object
            super().__init__(new_cpp_object)

            # 5. Store your custom properties
            self._orig_g = val

        def pack(self):
            if self._pack_expression is not None:
                self.g.interpolate_expr(self._pack_expression._cpp_object, None, None)

        @property
        def _original_arg(self):

            if isinstance(
                self._orig_g,
                (
                    dolfinx.cpp.fem.Function_complex128,
                    dolfinx.cpp.fem.Function_complex64,
                    dolfinx.cpp.fem.Function_float32,
                    dolfinx.cpp.fem.Function_float64,
                ),
            ):
                return dolfinx.fem.Function(self._ufl_space, self._orig_g.x, name=f"orig_{self._orig_g.name:s}")
            elif isinstance(
                self._orig_g,
                (
                    dolfinx.cpp.fem.Constant_complex64,
                    dolfinx.cpp.fem.Constant_complex128,
                    dolfinx.cpp.fem.Constant_float32,
                    dolfinx.cpp.fem.Constant_float64,
                ),
            ):
                return dolfinx.fem.Constant(
                    self._ufl_space.ufl_domain(), self._orig_g.value
                )
            return self._orig_g

        def reconstruct(self, V, g):
            return DirichletBC(self, new_value=g, V=V)

    def bc2space(bc, V):
        return get_sub(V, bc.function_space.component())

    def stage2spaces4bc(bc, V, Vbig, i):
        """used to figure out how to apply Dirichlet BC to each stage"""
        comp = bc.function_space.component()
        Vbig_i = Vbig.ufl_sub_spaces()[i]
        return get_sub(Vbig_i, comp)

    def extract_bcs(bcs: typing.Any) -> tuple[typing.Any]:
        """Extract boundary conditions"""
        new_bcs = []
        for bc in bcs:
            new_bcs.append(DirichletBC(bc))
        return new_bcs

    def create_variational_problem(F, u, bcs=None, aP=None, **kwargs):
        """Create a variational problem."""
        rank = len(np.unique([arg.number() for arg in F.arguments()]))
        if isinstance(u, ListTensor):
            u = u.subfunctions
        if rank == 2:
            a, L = ufl.system(F)
            a = ufl.extract_blocks(a)
            L = ufl.extract_blocks(L)
            return LinearProblem(
                a,
                L,
                u,
                bcs=bcs,
                petsc_options_prefix="IrkSomeLinearSolver",
                P=aP,
                **kwargs,
            )
        elif rank == 1:
            F = ufl.extract_blocks(F)
            return NonlinearProblem(
                F,
                u,
                petsc_options_prefix="IrkSomeNonlinearSolver",
                bcs=bcs,
                petsc_options=kwargs.get("solver_parameters"),
            )
        else:
            raise RuntimeError(f"Forms of rank {rank} are not supported in create_variational_problem")

    def create_variational_solver(
        problem: dolfinx.fem.petsc.LinearProblem | dolfinx.fem.petsc.NonlinearProblem,
        **kwargs,
    ):
        """Create a variational solver that uses PETSc SNES or KSP."""
        solver_parameters = kwargs.get("solver_parameters", {})
        solver = problem.solver
        solver_prefix = problem.solver.getOptionsPrefix()
        opts = PETSc.Options()
        opts.prefixPush(solver_prefix)
        for k, v in solver_parameters.items():
            opts.setValue(k, v)
        solver.setFromOptions()
        opts.prefixPop()
        # For some strange reason delValue doesn't respect prefixes
        for k, v in solver_parameters.items():
            opts.delValue(f"{solver_prefix}{k}")
        return problem

    def get_function_space(u: ufl.Coefficient | ufl.Argument) -> ufl.FunctionSpace:
        if isinstance(u, (ufl.Coefficient, ufl.Argument)):
            return u.ufl_function_space()
        else:
            raise ValueError(f"Cannot get function space for object of type {type(u)}")

    def get_stages(V: dolfinx.fem.FunctionSpace, num_stages: int) -> ListTensor:
        """
        Given a function space for a single time-step, get a duplicate of this space,
        repeated `num_stages` times.

        Args:
            V: Space for single step
            num_stages: Number of stages

        Returns:
            A coefficient in the new function space
        """
        _Vbig = [V.clone() for _ in range(num_stages)]
        Vbig = ufl.MixedFunctionSpace(*_Vbig)
        return ListTensor(*[dolfinx.fem.Function(Vi, name=f"stage_{i}") for i, Vi in enumerate(Vbig.ufl_sub_spaces())])

    class FloatConstantFunction(dolfinx.fem.Function):
        def __float__(self):
            if len(self.x.array) != 1:
                raise ValueError("Can only convert a FloatConstantFunction to float if it has exactly one degree of freedom")
            return float(self.x.array[0])

        def assign(self, value):
            self.x.array[0] = self.x.array.dtype.type(value)

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
            v = FloatConstantFunction(self.V)
            v.x.array[:] = val
            return v

    def get_mesh_constant(MC: MeshConstant | None) -> ufl.core.expr.Expr:
        return MC.Constant if MC is not None else ufl.constantvalue.ComplexValue

    def norm(
        v: ufl.core.expr.Expr, norm_type: str = "L2", mesh: ufl.Mesh | None = None
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

    def assemble(expr: ufl.core.expr.Expr | float):
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
    TrialFunction = lambda function_space: ufl.as_tensor(ufl.TrialFunctions(function_space))

    def Function(V: ufl.FunctionSpace | ufl.MixedFunctionSpace, name=None):
        """Create a function in the backend language."""
        if isinstance(V, ufl.MixedFunctionSpace):
            return ListTensor(*[dolfinx.fem.Function(Vi, name=f"{name}_{i}") for i, Vi in enumerate(V.ufl_sub_spaces())])
        else:
            return dolfinx.fem.Function(V, name=name)

    TestFunction = lambda function_space: ufl.as_tensor(ufl.TestFunctions(function_space))

    class Constant(ufl.constantvalue.ScalarValue):
        # NOTE: If dolfinx ever get's meshless constants we should change this
        def assign(self, value):
            self._value = value

    class EquationBCSplit:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("DOLFINx does not support EquationBCSplit")

    class EquationBC:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("DOLFINx does not support EquationBC")

    def invalidate_jacobian(solver: dolfinx.fem.petsc.LinearProblem):
        """Invalidate the Jacobian matrix in the backend language."""
        pass
        # raise RuntimeError("DOLFINx does not support Jacobian invalidation")

    def create_bounds_constrained_bc(V, g, sub_domain, bounds, solver_parameters=None):
        raise NotImplementedError(
            "Bounds-constrained BCs are not implemented for DOLFINx"
        )

    def getNullspace(V, Vbig, num_stages, nullspace):
        """
        Computes the nullspace for a multi-stage method.

        :arg V: The :class:`ufl.FunctionSpace` on which the original time-dependent PDE is posed.
        :arg Vbig: The multi-stage :class:`ufl.MixedFunctionSpace` for the stage problem
        :arg num_stages: The number of stages in the RK method
        :arg nullspace: The nullspace for the original problem.

        On output, we produce a PETSc nullspace defining the nullspace for the multistage problem.
        """
        if nullspace is None:
            nspnew = None
        else:
            raise NotImplementedError("Nullspace computation is not implemented for DOLFINx")
        return nspnew

except ModuleNotFoundError:
    pass
