"""DOLFINx backend for Irksome"""

try:
    import basix.ufl
    import dolfinx.fem.petsc
    import ufl
    import typing
    import numpy as np

    TestFunction = ufl.TestFunction

    def get_stage_space(V: ufl.FunctionSpace, num_stages:int)->ufl.FunctionSpace:
        if num_stages == 1:
            me = V.ufl_elemet()
        else:
            el = V.ufl_element()
            if el.num_sub_elements > 0:
                me = basix.ufl.mixed_element(np.tile(el.sub_elements, num_stages).tolist())
            else:
                me = basix.ufl.blocked_element(el, shape=(num_stages, ))
        return dolfinx.fem.functionspace(V.mesh, me)

    def extract_bcs(bcs: typing.Any)->tuple[typing.Any]:
        """Extract boundary conditions"""
        return bcs

    def create_nonlinearvariational_problem(F: ufl.Form, g: ufl.Coefficient, solver_parameters: dict):
        """Create a non-linear variational solver that uses PETSc SNES."""
        return dolfinx.fem.petsc.NonlinearProblem(F, g, petsc_options_prefix="IrkSomeSolver",
                                                     petsc_options=solver_parameters)

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
                import scifem
            except ModuleNotFoundError:
                raise RuntimeError("Scifem is required to make mesh-constants")

            self.V = scifem.create_real_functionspace(msh, ())

        def Constant(self, val=0.0) -> ufl.Coefficient:
            v = dolfinx.fem.Function(self.V)
            v.value = val
            return v

    def get_mesh_constant(MC: MeshConstant | None) -> ufl.core.expr.Expr:
        return MC.Constant if MC is not None else ufl.constantvalue.ComplexValue

except ModuleNotFoundError:
    pass
