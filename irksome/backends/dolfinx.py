"""DOLFINx backend for Irksome"""

from __future__ import annotations
from collections.abc import Sequence
from functools import singledispatchmethod

from irksome.tools import get_sub

from mpi4py import MPI
from petsc4py import PETSc

try:
    import dolfinx.fem.petsc
except ModuleNotFoundError:
    raise ImportError(
        "DOLFINx is not installed. Please install DOLFINx to use the DOLFINx backend of Irksome."
    )

from ufl.corealg.dag_traverser import DAGTraverser
import ufl
import typing
import numpy as np
import numpy.typing as npt


def extract_dtype(expr: ufl.core.expr.Expr) -> npt.DTypeLike:
    """Extract the dtype from an expression.

    Looks for any constants or coefficients and returning their dtype.
    This is necessary for determining which DOLFINx DirichletBC constructor
    to use when packing UFL expressions into DOLFINx Expressions for use in
    BC reconstruction.
    """
    consts = ufl.algorithms.analysis.extract_constants(expr)
    for c in consts:
        if hasattr(c, "dtype"):
            return c.dtype
    coeffs = ufl.algorithms.extract_coefficients(expr)
    for c in coeffs:
        if hasattr(c, "dtype"):
            return c.dtype
    raise ValueError(
        "Could not extract dtype from expression, please ensure that all constants and coefficients have a dtype attribute"
    )


try:
    from ufl.algorithms.extract_linear_combination import extract_linear_combination
except ImportError:

    class LinearCombinationExtractor(DAGTraverser):
        """Bottom-up DAG traverser for extracting linear combinations.

        To process an arbitrary mathematical expression, this traverser categorizes
        every node in the DAG into one of two states, returning different types for each:

        1. Scalar Weights (Returns: {py:class}`ufl.core.expr.Expr`)
        If a node and all its children represent a global scalar value (e.g.,
        {py:class}`ufl.FloatValue`, {py:class}`ufl.Constant`), the traverser
        propagates the actual UFL expression upwards. It does not evaluate them
        to Python floats, preserving the full UFL AST of the constants.

        2. Spatial Fields (Returns: `list[tuple[ufl.core.expr.Expr, ufl.Coefficient]]`)
        If a node contains spatial functions (standard Coefficients), it must
        maintain the strict algebraic structure of a linear combination. Therefore,
        it returns a list of `(weight, function)` tuples, where `weight` is the
        accumulated UFL expression and `function` is the base spatial field.

        By strictly distinguishing between the two (checking `isinstance(..., list)`),
        the traverser can safely apply algebraic rules (e.g., multiplying a list by a
        scalar weight expression distributes the weight) and instantly catch illegal
        non-linear operations (e.g., attempting to multiply two lists together).
        """

        def __init__(self, **kwargs):
            """Initialize LinearCombinationExtractor with memoization and no compression.

            Compression is disabled to avoid hashing unhashable return types (like lists)
            while preserving the `_visited_cache` memoization.
            """
            kwargs["compress"] = False
            super().__init__(**kwargs)

        @singledispatchmethod
        def process(self, o: ufl.classes.Expr, **kwargs):
            """Fallback for any unsupported node types."""
            raise ValueError(
                f"Unsupported UFL node type for linear combinations: {type(o)}"
            )

        # ---------------------------------------------------------
        # 1. Terminals (Leaves) - No children to evaluate
        # ---------------------------------------------------------
        @process.register(ufl.classes.IntValue)
        @process.register(ufl.classes.FloatValue)
        @process.register(ufl.classes.ScalarValue)
        def _(self, o, **kwargs):
            # Return the UFL expression itself
            return o

        @process.register(ufl.classes.Zero)
        def _(self, o, **kwargs):
            return o if o.ufl_shape == () else []

        @process.register(ufl.Constant)
        def _(self, o, **kwargs):
            if o.ufl_shape == ():
                return o
            raise ValueError(
                f"Only scalar constants are supported, got shape {o.ufl_shape}"
            )

        @process.register(ufl.Matrix)
        @process.register(ufl.coefficient.BaseCoefficient)
        def _(self, o, **kwargs):
            # Check for real-valued elements
            if isinstance(
                o, ufl.core.expr.Expr
            ) and ufl.checks.is_scalar_constant_expression(o):
                return o
            return [(ufl.as_ufl(1.0), o)]

        # ---------------------------------------------------------
        # 2. Operators - Use @postorder to evaluate operands first
        # ---------------------------------------------------------
        @process.register(ufl.classes.Sum)
        @DAGTraverser.postorder
        def _(self, o, *operands, **kwargs):
            # If no operands are lists, this is a pure scalar addition.
            # We construct a new UFL expression by safely summing them.
            if all(not isinstance(op, list) for op in operands):
                res = operands[0]
                for op in operands[1:]:
                    res = res + op
                return res

            # Otherwise, accumulate the spatial functions
            res = []
            for op_res in operands:
                if isinstance(op_res, list):
                    res.extend(op_res)
                else:
                    raise ValueError(
                        "Cannot directly add a raw scalar expression to a spatial function."
                    )
            return res

        @process.register(ufl.Action)
        def _(self, o, **kwargs):
            # An Action node represents a matrix-vector product (e.g., A * u).
            # This cannot be reduced to a simple algebraic linear combination of arrays.
            raise ValueError(
                "Non-linear expression detected: product of two spatial functions."
            )

        @process.register(ufl.classes.FormSum)
        @process.register(ufl.form.FormSum)
        def _(self, o, **kwargs):
            res = []
            components = o.components()
            weights = o.weights()
            for weight, comp in zip(weights, components):
                # Evaluate the base component (e.g., Matrix or Cofunction)
                comp_res = self(comp, **kwargs)

                # Evaluate the weight (in case it contains sub-expressions)
                w_res = (
                    self(weight, **kwargs)
                    if isinstance(weight, ufl.classes.Expr)
                    else weight
                )

                if isinstance(comp_res, list):
                    # Distribute this FormSum weight into the component's linear combination
                    res.extend([(w_res * w, f) for w, f in comp_res])
                else:
                    raise ValueError(
                        "Cannot directly add a raw scalar expression to a spatial function."
                    )

            return res

        @process.register(ufl.classes.Product)
        @DAGTraverser.postorder
        def _(self, o, *operands, **kwargs):
            op1_res, op2_res = operands
            # Each of the operands are either a scalar UFL expression (float, Constant, etc.)
            # or a list of (weight, function) tuples.
            # The following cases are possible:
            # 1. Both operands are scalars: return the product of the two UFL expressions.
            # 2. One operand is a scalar, the other is a list: distribute the scalar across the list.
            # 3. Both operands are lists: this is a non-linear operation and should raise an error.
            is_list1 = isinstance(op1_res, list)
            is_list2 = isinstance(op2_res, list)
            if not is_list1 and not is_list2:
                return op1_res * op2_res  # UFL operator overloading takes over
            elif not is_list1 and is_list2:
                return [(op1_res * w, f) for w, f in op2_res]
            elif not is_list2 and is_list1:
                return [(op2_res * w, f) for w, f in op1_res]
            else:
                raise ValueError(
                    "Non-linear expression detected: product of two spatial functions."
                )

        @process.register(ufl.classes.Division)
        @DAGTraverser.postorder
        def _(self, o, *operands, **kwargs):
            num_res, den_res = operands
            if isinstance(den_res, list):
                raise ValueError(
                    "Non-linear expression detected: division by a spatial function."
                )

            if not isinstance(num_res, list):
                return num_res / den_res
            return [(w / den_res, f) for w, f in num_res]

        @process.register(ufl.classes.Power)
        @DAGTraverser.postorder
        def _(self, o, *operands, **kwargs):
            base_res, exp_res = operands
            if isinstance(base_res, list) or isinstance(exp_res, list):
                raise ValueError(
                    "Non-linear expression detected: power involving a spatial function."
                )
            return base_res**exp_res

        # ---------------------------------------------------------
        # 3. Tensor Components (Vector-Scalar multiplication)
        # ---------------------------------------------------------
        @process.register(ufl.classes.MultiIndex)
        def _(self, o, **kwargs):
            return o

        @process.register(ufl.classes.Indexed)
        @DAGTraverser.postorder
        def _(self, o, *operands, **kwargs):
            base_res, indices_res = operands
            
            # We ONLY allow UFL's free indices. Fixed components (FixedIndex) 
            # or slices (SliceIndex) do not represent the whole operator.
            if not all(isinstance(idx, ufl.classes.Index) for idx in indices_res.indices()):
                raise ValueError(
                    "Explicit component indexing (e.g., u[0]) is not supported. "
                    "Only full-operator free indices are allowed."
                )
            
            if isinstance(base_res, list):
                # Temporarily return the Indexed spatial field.
                # The ComponentTensor node higher up the tree will collapse this 
                # back into the un-indexed base operator using remove_component_tensors.
                return [(w, ufl.classes.Indexed(f, indices_res)) for w, f in base_res]
            
            return ufl.classes.Indexed(base_res, indices_res)

        @process.register(ufl.classes.ComponentTensor)
        @DAGTraverser.postorder
        def _(self, o, *operands, **kwargs):
            from ufl.algorithms.remove_component_tensors import remove_component_tensors
            
            expr_res, indices_res = operands
            
            if isinstance(expr_res, list):
                res = []
                for w, f in expr_res:
                    # f is something like Indexed(stage_0, i). Reconstruct the tensor:
                    reconstructed_func = ufl.classes.ComponentTensor(f, indices_res)
                    
                    # Collapse it back to the base operator (e.g., stage_0)
                    simplified_func = remove_component_tensors(reconstructed_func)
                    
                    # Enforce that the result actually resolved back to a whole operator
                    if isinstance(simplified_func, (ufl.classes.Indexed, ufl.classes.ComponentTensor)):
                        raise ValueError(
                            "Indexing did not cleanly resolve to a complete spatial operator."
                        )
                        
                    res.append((w, simplified_func))
                return res
                
            return ufl.classes.ComponentTensor(expr_res, indices_res)

    def extract_linear_combination(
        expr: ufl.core.expr.Expr | ufl.form.BaseForm,
    ) -> list[tuple[ufl.core.expr.Expr, ufl.coefficient.BaseCoefficient]]:
        """Wrapper to initialize traverser and extract linear combinations.

        Returns:
            A list of tuples where the first element is the UFL expression of the
            weight, and the second element is the base UFL Coefficient (spatial function).
        """
        extractor = LinearCombinationExtractor()
        final_result = extractor(expr)

        if not isinstance(final_result, list):
            raise ValueError(
                "Expression evaluated to a pure scalar, no spatial functions found."
            )

        return final_result


def function_iadd(func: "dolfinx.fem.Function", expr: ufl.core.expr.Expr) -> None:
    """Add a UFL expression to a DOLFINx Function in-place (func += expr).

    Extracts the linear combination structure and adds terms directly using
    PETSc vector operations. Works with mixed spaces and subfunction views.

    :param func: DOLFINx Function or subfunction view to update in-place
    :param expr: UFL expression (linear combination of Functions)

    .. code-block:: python

        function_iadd(u, 0.5 * v + 0.3 * w)  # equivalent to u += 0.5*v + 0.3*w
    """
    # Extract the linear combination: [(weight1, func1), (weight2, func2), ...]
    terms = extract_linear_combination(expr)

    # Add each term using array operations (works with mixed spaces when dofmaps match)
    for weight, term_func in terms:
        func.x.array[:] += float(weight) * term_func.x.array[:]


# Patching of DOLFINx objects to mimick firedrake naming and properties.


def function_space_length(self):
    return 1


def mixed_space_length(self):
    return len(self.ufl_sub_spaces())


def function_subfunctions(self):
    """Get subfunctions for a DOLFINx function, which may be in a mixed space.

    This function is called when updating `u0` in time-stepping problems.
    """
    return [self]


def function_iadd_method(self, other):
    """In-place addition for DOLFINx functions (self += other).

    This method is monkey-patched onto :py:class:`dolfinx.fem.Function` to enable
    Firedrake-style arithmetic operations.

    :param self: The function to be modified
    :param other: The expression to add. Can be a UFL expression, Function, Constant, or scalar.

    :returns: self (for chaining)
    """
    # Delegate to the standalone function_iadd which handles all the complexity
    function_iadd(self, other)
    return self


dolfinx.fem.FunctionSpace.__len__ = function_space_length
dolfinx.fem.Function.subfunctions = property(function_subfunctions)
dolfinx.fem.Function.__iadd__ = function_iadd_method
ufl.MixedFunctionSpace.__len__ = mixed_space_length


class ListTensor(ufl.tensors.ListTensor):
    """A list tensor that exposes subfunctions for DOLFINx functions"""

    @property
    def subfunctions(self):
        sub_funcs = []
        for i in range(self.ufl_shape[0]):
            func = self.ufl_operands[i]
            sub_funcs.append(func)
        return sub_funcs

    def function_space(self):
        return ufl.MixedFunctionSpace(*[self.ufl_operands[i].ufl_function_space() for i in range(self.ufl_shape[0])])


class LinearProblem(dolfinx.fem.petsc.LinearProblem):
    """Overloaded linear problem that pack BCs before solving"""

    def solve(self, bounds=None):
        if bounds is not None:
            raise NotImplementedError("Bounds-constrained solves are not implemented for DOLFINx")
        [bc.pack() for bc in self._bcs]
        super().solve()


class NonlinearProblem(dolfinx.fem.petsc.NonlinearProblem):
    """Overloaded nonlinear problem that pack BCs before solving.

    Does each  Newton iteration or line search step by overriding the
    SNES function and Jacobian assembly routines to pack BCs before assembly.
    """

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
        self._bcs = bcs

        def assemble_residual(
            _snes: PETSc.SNES,  # type: ignore[name-defined]
            x: PETSc.Vec,  # type: ignore[name-defined]
            b: PETSc.Vec,  # type: ignore[name-defined]
            u: dolfinx.fem.Function | Sequence[dolfinx.fem.Function],
            residual: dolfinx.fem.Form | Sequence[dolfinx.fem.Form],
            jacobian: dolfinx.fem.Form | Sequence[Sequence[dolfinx.fem.Form]],
            bcs: Sequence[dolfinx.fem.DirichletBC],
            _blocks: tuple[tuple[int, int, int], ...] | None = None,
        ):

            [bc.pack() for bc in bcs]
            # Cover single stage problems where _blocks is None
            dolfinx.fem.petsc.assemble_residual(
                _snes, x, b, u, residual, jacobian, bcs, _blocks
            )

        if isinstance(u, Sequence) and len(u) == 1:
            assert len(self.F) == 1 and len(self.J) == 1 and len(self.J[0]) == 1
            kargs = {
                "u": u[0],
                "residual": self.F[0],
                "jacobian": self.J[0][0],
                "bcs": self._bcs,
                "_blocks": self.b.getAttr("_blocks"),
            }
            jac_kargs = {
                "u": u[0],
                "jacobian": self.J[0][0],
                "bcs": self._bcs,
                "preconditioner": self._preconditioner,
            }
            if self._P_mat is not None:
                assert (
                    len(self._preconditioner) == 1
                    and len(self._preconditioner[0]) == 1
                )
        else:
            kargs = {
                "u": u,
                "residual": self.F,
                "jacobian": self.J,
                "bcs": self._bcs,
                "_blocks": self.b.getAttr("_blocks"),
            }
            jac_kargs = {
                "u": u,
                "jacobian": self.J,
                "bcs": self._bcs,
                "preconditioner": self._preconditioner,
            }
        self.solver.setFunction(assemble_residual, self.b, kargs=kargs)

        def assemble_jacobian(
            _snes: PETSc.SNES,  # type: ignore[name-defined]
            x: PETSc.Vec,  # type: ignore[name-defined]
            J: PETSc.Mat,  # type: ignore[name-defined]
            P_mat: PETSc.Mat,  # type: ignore[name-defined]
            u: Sequence[dolfinx.fem.Function] | dolfinx.fem.Function,
            jacobian: dolfinx.fem.Form | Sequence[Sequence[dolfinx.fem.Form]],
            preconditioner: dolfinx.fem.Form
            | Sequence[Sequence[dolfinx.fem.Form]]
            | None,
            bcs: Sequence[dolfinx.fem.DirichletBC],
        ):
            [bc.pack() for bc in bcs]
            dolfinx.fem.petsc.assemble_jacobian(
                _snes, x, J, P_mat, u, jacobian, preconditioner, bcs
            )

        self.solver.setJacobian(
            assemble_jacobian, self.A, self.P_mat, kargs=jac_kargs
        )

    def solve(self, bounds=None):
        if bounds is not None:
            raise NotImplementedError("Bounds-constrained solves are not implemented for DOLFINx")
        super().solve()

    @property
    def snes(self) -> PETSc.SNES:  # type: ignore[name-defined]
        return self.solver


def get_stage_space(V: ufl.FunctionSpace, num_stages: int) -> ufl.FunctionSpace:
    if num_stages == 1:
        space_list = [V]
    else:
        space_list = [V.clone() for _ in range(num_stages)]
    return ufl.MixedFunctionSpace(*space_list)


class DirichletBC(dolfinx.fem.DirichletBC):
    _pack_expression: dolfinx.fem.Expression | None
    _ufl_expr: ufl.core.expr.Expr | None  # Store original UFL expression

    def __init__(self, g: ufl.core.expr.Expr, dofs: npt.NDArray[np.int32], V: dolfinx.fem.FunctionSpace):
        """
        Create an Irksome compatible DirichletBC from an existing DOLFINx bc.

        .. note::
            This is not a user-facing class, but rather an internal class used for BC reconstruction in time-stepping problems.
            Use: :func:`irksome.backends.dolfinx.dirichletbc`.

        :param g: The boundary condition expression
        :param dofs: An array of degree-of-freedom indices in V
        :param V: The space to construct the BC on.
        """

        # Store original UFL expression for time-varying BCs
        if not isinstance(g, (dolfinx.fem.Function, dolfinx.fem.Constant, int, float, complex)):
            self._ufl_expr = g  # Save the symbolic expression
        else:
            self._ufl_expr = None

        # If reconstructing with a sub space, we need to get the subspace dof indices
        # If working with a subspace of a single stage, we need to create the (parent_dof, sub_dof) mapping
        if V.component() != []:
            V_sub, sub_to_parent = V.collapse()
            if len(sub_to_parent) != 1:
                raise NotImplementedError(
                    "Mixed topology is not supported for reconstructing BCs with UFL expressions"
                )
            else:
                sub_to_parent = sub_to_parent[0]
                parent_to_sub = np.full(
                    (V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)
                    * V.dofmap.index_map_bs,
                    -1,
                    dtype=np.int32,
                )
                parent_to_sub[sub_to_parent] = np.arange(len(sub_to_parent))
                sub_dofs = parent_to_sub[dofs]
                dofs = (dofs, sub_dofs)

        # If we are not reconstructing the BC with a new value, we can reuse existing C++ objects
        self._pack_expression = None

        # If we are reconstructing the BC with a new value, we need to check if the new value is a DOLFINx function or Constant.
        # If True, we do not need to do anything for reconstruction.
        if isinstance(g, (dolfinx.fem.Function, dolfinx.fem.Constant)):
            val = g
            self._pack_expression = None
        else:
            # If not, we need to take the ufl.core.expr.Expr and pack it into a DOLFINx Expression
            if V.component() != []:
                val = dolfinx.fem.Function(V_sub, name=f"bc_{str(g)}")._cpp_object
            else:
                val = dolfinx.fem.Function(V, name=f"bc_{str(g)}")._cpp_object
            self._pack_expression = dolfinx.fem.Expression(g, V.element.interpolation_points)

        # Get correct C++ implementation based on dtype of expression
        dtype = extract_dtype(g)
        if np.issubdtype(dtype, np.float32):
            bctype = dolfinx.cpp.fem.DirichletBC_float32
        elif np.issubdtype(dtype, np.float64):
            bctype = dolfinx.cpp.fem.DirichletBC_float64
        elif np.issubdtype(dtype, np.complex64):
            bctype = dolfinx.cpp.fem.DirichletBC_complex64
        elif np.issubdtype(dtype, np.complex128):
            bctype = dolfinx.cpp.fem.DirichletBC_complex128
        else:
            raise NotImplementedError(f"Type {dtype} not supported.")

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
            and val.function_space == V._cpp_object
        ):
            new_cpp_object = bctype(val, dofs)
        else:
            # Depending on your FEniCSx version, the C++ constructor might strictly
            # expect the C++ FunctionSpace instead of the Python FunctionSpace wrapper.
            try:
                new_cpp_object = bctype(val, dofs, V._cpp_object)
            except TypeError:
                new_cpp_object = bctype(val._cpp_object, dofs, V._cpp_object)

        # 4. Initialize the parent dolfinx.fem.DirichletBC wrapper with the newly minted C++ object
        try:
            super().__init__(new_cpp_object)
            # Attach UFL function space (to be able to reconstruct functions and constants on the same UFL domain)
            self._ufl_space = V.ufl_function_space()
            self._orig_g = val
        except TypeError:
            # In newer versions of DOLFINx, both the V and the original expression is
            # required to construct the DirichletBC
            # Long-term we can remove _orig_g and _ufl_space and _ufl_space and use
            # bc.function_space and bc.g
            super().__init__(new_cpp_object, V=V, g=val)
            self._ufl_space = self.function_space
            self._orig_g = self.g

    def pack(self):
        if self._pack_expression is not None:
            self.g.interpolate_expr(self._pack_expression._cpp_object, None, None)

    @property
    def _original_arg(self):
        # If we stored the original UFL expression, return it for time substitution
        if hasattr(self, "_ufl_expr") and self._ufl_expr is not None:
            return self._ufl_expr

        # Otherwise return the wrapped Function/Constant
        if isinstance(
            self._orig_g,
            (
                dolfinx.cpp.fem.Function_complex128,
                dolfinx.cpp.fem.Function_complex64,
                dolfinx.cpp.fem.Function_float32,
                dolfinx.cpp.fem.Function_float64,
            ),
        ):
            return dolfinx.fem.Function(self._ufl_space, dolfinx.la.Vector(self._orig_g.x), name=f"orig_{self._orig_g.name:s}")
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
        return DirichletBC(g, self.dof_indices()[0], V=V)


def dirichletbc(
    value: ufl.core.expr.Expr,
    dofs: npt.NDArray[np.int32],
    V: dolfinx.fem.FunctionSpace | None = None,
) -> DirichletBC:
    """Overloaded DirichletBC so that we can reconstruct BCs with UFL expressions.

    .. note::
        This class is user-facing.

    :param value: A UFL expression representing the boundary condition.
    :param dofs: An array of degree-of-freedom indices in `V` where the BC should be applied.
    :param V: The function space on which the BC applies. It can be a subspace of a mixed/blocked space.
    """
    return DirichletBC(value, dofs, V)


def bc2space(bc, V):
    return get_sub(V, bc.function_space.component())


def stage2spaces4bc(bc, V, Vbig, i):
    """used to figure out how to apply Dirichlet BC to each stage"""
    comp = bc.function_space.component()
    Vbig_i = Vbig.ufl_sub_spaces()[i]
    return get_sub(Vbig_i, comp)


def extract_bcs(bcs: typing.Any) -> tuple[typing.Any]:
    """Extract boundary conditions"""
    return bcs


def create_variational_problem(F, u, bcs=None, aP=None, **kwargs):
    """Create a variational problem."""
    rank = len(np.unique([arg.number() for arg in F.arguments()]))
    prefix = kwargs.get("petsc_options_prefix", "IrkSomeSolver")
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
            petsc_options_prefix=prefix,
            P=aP,
            **kwargs,
        )
    elif rank == 1:
        F = ufl.extract_blocks(F)
        return NonlinearProblem(
            F,
            u,
            petsc_options_prefix=prefix,
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

    # Update solver prefix
    solver_prefix = solver_parameters.get(
        "petsc_options_prefix", problem.solver.getOptionsPrefix()
    )
    problem._petsc_options_prefix = solver_prefix
    problem.solver.setOptionsPrefix(solver_prefix)
    problem.A.setOptionsPrefix(f"{solver_prefix}A_")
    problem.b.setOptionsPrefix(f"{solver_prefix}b_")
    problem.x.setOptionsPrefix(f"{solver_prefix}x_")
    if problem.P_mat is not None:
        problem.P_mat.setOptionsPrefix(f"{solver_prefix}P_mat_")
    # Push new options with prefix, filtering out irksome-internal keys
    petsc_opts = {
        k: v for k, v in solver_parameters.items() if k != "petsc_options_prefix"
    }
    opts = PETSc.Options()
    opts.prefixPush(solver_prefix)
    for k, v in petsc_opts.items():
        opts.setValue(k, v)
    solver.setFromOptions()
    opts.prefixPop()
    # For some strange reason delValue doesn't respect prefixes
    for k in petsc_opts:
        opts.delValue(f"{solver_prefix}{k}")

    nullspace = kwargs.get("nullspace", None)
    near_nullspace = kwargs.get("near_nullspace", None)
    if nullspace is not None:
        problem.A.setNullSpace(nullspace)
    if near_nullspace is not None:
        problem.A.setNearNullSpace(near_nullspace)
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

    :param V: Space for single step
    :param num_stages: Number of stages
    :returns: A coefficient in the new function space
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
        self.x.scatter_forward()


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
        dx = ufl.Measure("dx", domain=mesh)
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
    norm_global = form.mesh.comm.allreduce(norm_loc, op=MPI.SUM)
    return norm_global ** (1 / p)


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


def TrialFunction(function_space):
    """Create a trial function in the backend language.

    This is required as :func:`ufl.TrialFunctions` returns a tuple of trial functions,
    which we need to convert to a tensor for consistency with the Firedrake backend.
    """
    return ufl.as_tensor(ufl.TrialFunctions(function_space))


def Function(V: ufl.FunctionSpace | ufl.MixedFunctionSpace, name=None):
    """Create a function in the backend language."""
    if isinstance(V, ufl.MixedFunctionSpace):
        return ListTensor(
            *[
                dolfinx.fem.Function(Vi, name=f"{name}_{i}")
                for i, Vi in enumerate(V.ufl_sub_spaces())
            ]
        )
    else:
        return dolfinx.fem.Function(V, name=name)


def TestFunction(function_space):
    """Create a test function in the backend language.

    This is required as :func:`ufl.TestFunctions` returns a tuple of test functions,
    which we need to convert to a tensor for consistency with the Firedrake backend.
    """
    return ufl.as_tensor(ufl.TestFunctions(function_space))


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


def create_bounds_constrained_bc(V, g, sub_domain, bounds, solver_parameters=None):
    raise NotImplementedError(
        "Bounds-constrained BCs are not implemented for DOLFINx"
    )


def getNullspace(V, Vbig, num_stages, nullspace):
    """
    Computes the nullspace for a multi-stage method.

    :param V: The :class:`ufl.FunctionSpace` on which the original time-dependent PDE is posed.
    :param Vbig: The multi-stage :class:`ufl.MixedFunctionSpace` for the stage problem
    :param num_stages: The number of stages in the RK method
    :param nullspace: The nullspace for the original problem.

    On output, we produce a PETSc nullspace defining the nullspace for the multistage problem.
    """
    if nullspace is None:
        nspnew = None
    else:
        old_vecs = nullspace.getVecs()
        tmp_func = dolfinx.fem.Function(V)
        new_vecs = [
            dolfinx.fem.petsc.create_vector(Vbig.ufl_sub_spaces())
            for _ in range(len(old_vecs))
        ]
        for i in range(len(old_vecs)):
            # Copy the old nullspace vector into all stages of the new nullspace vector
            old_vecs[i].copy(tmp_func.x.petsc_vec)
            tmp_func.x.scatter_forward()
            dolfinx.fem.petsc.assign(
                [tmp_func for _ in range(num_stages)], new_vecs[i]
            )
        nspnew = PETSc.NullSpace().create(
            vectors=new_vecs, comm=nullspace.getComm()
        )
    return nspnew


derivative = ufl.derivative
