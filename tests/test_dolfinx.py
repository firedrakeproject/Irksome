import pytest
import gc
import logging

from mpi4py import MPI
from petsc4py import PETSc

dolfinx = pytest.importorskip("dolfinx")

import basix.ufl
import numpy as np
import ufl

from ufl import (
    div,
    grad,
    inner,
    dx,
    as_vector,
    split,
    SpatialCoordinate,
    TestFunction,
    TestFunctions,
    TrialFunction,
)

from irksome import GaussLegendre, Dt, MeshConstant
from irksome.backends.dolfinx import assemble, dirichletbc, norm
from irksome.stage_derivative import StageDerivativeTimeStepper
from irksome.tools import AI


@pytest.mark.parametrize("num_stages", [1, 2, 3, 4])
def test_stokes(num_stages):
    butcher_tableau = GaussLegendre(num_stages=num_stages)

    N = 13
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(msh.topology)

    el_u = basix.ufl.element("Lagrange", msh.basix_cell(), 3, shape=(msh.geometry.dim,))
    el_p = basix.ufl.element("Lagrange", msh.basix_cell(), 2)
    el_me = basix.ufl.mixed_element([el_u, el_p])
    W = dolfinx.fem.functionspace(msh, el_me)

    MC = MeshConstant(msh, backend="dolfinx")
    t = MC.Constant(0.0)
    t.name = "t"
    dt = MC.Constant(1.0 / N)
    dt.name = "dt"
    (x, y) = SpatialCoordinate(msh)

    uexact = as_vector([x * t + y**2, -y * t + t * (x**2)])
    pexact = x + y * t ** (2 * num_stages)

    u_rhs = Dt(uexact) - div(grad(uexact)) + grad(pexact)
    p_rhs = -div(uexact)

    z = dolfinx.fem.Function(W)
    u, p = split(z)
    (v, q) = TestFunctions(W)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(p, div(v)) * dx
        - inner(div(u), q) * dx
        - inner(u_rhs, v) * dx
        - inner(p_rhs, q) * dx
    )

    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        W.sub(0), msh.topology.dim - 1, boundary_facets
    )

    msh.topology.create_connectivity(0, msh.topology.dim)
    corner_vertex = dolfinx.mesh.locate_entities(
        msh, 0, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0)
    )
    corner_dof = dolfinx.fem.locate_dofs_topological(W.sub(1), 0, corner_vertex)

    # Dirichlet BCs (irksome style with UFL expression)
    bc = dirichletbc(uexact, boundary_dofs, W.sub(0))
    bc_p = dirichletbc(pexact, corner_dof, W.sub(1))
    bcs = [bc, bc_p]

    # Initial conditions
    bc_expr = dolfinx.fem.Expression(uexact, W.sub(0).element.interpolation_points)
    z.sub(0).interpolate(bc_expr)
    z.sub(1).interpolate(
        dolfinx.fem.Expression(pexact, W.sub(1).element.interpolation_points)
    )

    solver_parameters = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_error_if_not_converged": True,
        "snes_max_iter": 50,
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
        "snes_linesearch_type": "none",
        "mat_mumps_icntl_24": 1,
        "mat_mumps_icntl_25": 0,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_monitor": None,
        "petsc_options_prefix": f"IrkSomeStokesSolver{num_stages}",
    }
    linear_stepper = StageDerivativeTimeStepper(
        F,
        butcher_tableau,
        t,
        dt,
        z,
        bcs=bcs,
        Fp=None,
        bc_type="DAE",
        splitting=AI,
        solver_parameters=solver_parameters,
        backend="dolfinx",
    )
    z0 = z.sub(0).collapse()

    _, vel_to_mixed = W.sub(0).collapse()

    end_time = 1.0
    while float(t) < end_time - 1e-10:
        if (
            float(t) + float(dt) > end_time
        ):  # To avoid floating point issues at end of simulation.
            dt.assign(end_time - float(t))
        linear_stepper.advance()
        t.assign(float(t) + float(dt))
        z0.x.array[:] = z.x.array[vel_to_mixed]
        z0.x.scatter_forward()

        error = norm(z0 - uexact, norm_type="L2", mesh=msh)
        assert error < 2e-10, f"Error {error} exceeds tolerance at timestep {float(t)}"
    msh.comm.Barrier()  # Ensure all processes have finished before cleaning up
    del linear_stepper
    msh.comm.Barrier()
    gc.collect()
    msh.comm.Barrier()


def test_default_solver_parameters():
    """Constructing a stepper without solver parameters uses PETSc defaults."""
    N = 4
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))

    MC = MeshConstant(msh, backend="dolfinx")
    t = MC.Constant(0.0)
    dt = MC.Constant(0.1)
    (x, y) = SpatialCoordinate(msh)
    uexact = t + x + y

    u = dolfinx.fem.Function(V)
    u.interpolate(dolfinx.fem.Expression(uexact, V.element.interpolation_points))
    v = TestFunction(V)
    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - inner(Dt(uexact), v) * dx

    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        V, msh.topology.dim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology)
    )
    bcs = [dirichletbc(uexact, boundary_dofs, V)]

    stepper = StageDerivativeTimeStepper(
        F, GaussLegendre(1), t, dt, u, bcs=bcs, backend="dolfinx"
    )
    stepper.advance()
    t.assign(float(t) + float(dt))

    error = norm(u - uexact, norm_type="L2", mesh=msh)
    assert error < 1e-5, f"Error {error} exceeds tolerance"


def test_assemble():
    """Backend assemble returns floats unchanged and assembles forms by rank."""

    assert assemble(1.5) == 1.5

    msh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    assert np.isclose(assemble(1.0 * dx(domain=msh)), 1.0)

    b = assemble(TestFunction(V) * dx)
    assert np.isclose(b.array.sum(), 1.0)


def _stage_jacobian(stepper):
    """Assemble the stage Jacobian of a DOLFINx-backed stepper."""
    problem = stepper.solver
    A = problem.A
    P = problem.P_mat if problem.P_mat is not None else A
    problem.snes.computeJacobian(problem.x, A, P)
    return A


@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_stokes_stage_nullspace(num_stages):
    """A constant pressure in each stage spans the Stokes stage nullspace."""
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    el_u = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
    el_p = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([el_u, el_p]))

    MC = MeshConstant(msh, backend="dolfinx")
    t = MC.Constant(0.0)
    dt = MC.Constant(0.1)
    (x, y) = SpatialCoordinate(msh)

    z = dolfinx.fem.Function(W)
    u, p = split(z)
    (v, q) = TestFunctions(W)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(p, div(v)) * dx
        - inner(div(u), q) * dx
    )

    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        W.sub(0),
        msh.topology.dim - 1,
        dolfinx.mesh.exterior_facet_indices(msh.topology),
    )
    bcs = [dirichletbc(as_vector([t * y, t * x]), boundary_dofs, W.sub(0))]

    # Constant pressure, zero velocity
    n = dolfinx.fem.Function(W)
    _, p_to_mixed = W.sub(1).collapse()
    n.x.array[p_to_mixed] = 1.0
    n.x.petsc_vec.normalize()
    nullspace = PETSc.NullSpace().create(vectors=[n.x.petsc_vec], comm=msh.comm)

    stepper = StageDerivativeTimeStepper(
        F,
        GaussLegendre(num_stages),
        t,
        dt,
        z,
        bcs=bcs,
        nullspace=nullspace,
        solver_parameters={"petsc_options_prefix": f"stokes_nsp_{num_stages}_"},
        backend="dolfinx",
    )
    A = _stage_jacobian(stepper)
    stage_nullspace = A.getNullSpace()
    assert len(stage_nullspace.getVecs()) == num_stages
    assert stage_nullspace.test(A)


@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_constant_stage_nullspace(num_stages):
    """A constant nullspace becomes one constant per stage."""
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))

    MC = MeshConstant(msh, backend="dolfinx")
    t = MC.Constant(0.0)
    dt = MC.Constant(0.1)

    u = dolfinx.fem.Function(V)
    v = TestFunction(V)
    F = inner(grad(Dt(u)), grad(v)) * dx + inner(grad(u), grad(v)) * dx

    nullspace = PETSc.NullSpace().create(constant=True, comm=msh.comm)
    stepper = StageDerivativeTimeStepper(
        F,
        GaussLegendre(num_stages),
        t,
        dt,
        u,
        nullspace=nullspace,
        solver_parameters={"petsc_options_prefix": f"constant_nsp_{num_stages}_"},
        backend="dolfinx",
    )
    A = _stage_jacobian(stepper)
    stage_nullspace = A.getNullSpace()
    assert len(stage_nullspace.getVecs()) == num_stages
    assert stage_nullspace.test(A)


def _manufactured_heat_problem():
    """Heat equation with a manufactured solution, flux BCs and its exact Jacobian.

    The solution is linear in time and quadratic in space, so Gauss-Legendre with P2
    reproduces it up to round-off. Natural BCs keep the stage operators free of
    Dirichlet rows.
    """
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 2))
    MC = MeshConstant(msh, backend="dolfinx")
    t = MC.Constant(0.0)
    dt = MC.Constant(0.1)
    (x, y) = SpatialCoordinate(msh)
    n = ufl.FacetNormal(msh)

    # f is built from the stepper's t, so that stage times are substituted in it
    t_var = ufl.variable(t)
    u_ex = (1 + t_var) * (x**2 + x * y)
    f = ufl.diff(u_ex, t_var) - div(grad(u_ex))

    u = dolfinx.fem.Function(V)
    u.interpolate(dolfinx.fem.Expression(u_ex, V.element.interpolation_points))
    v = TestFunction(V)
    w = TrialFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + inner(grad(u), grad(v)) * dx
        - inner(f, v) * dx
        - inner(ufl.dot(grad(u_ex), n), v) * ufl.ds
    )
    J = inner(Dt(w), v) * dx + inner(grad(w), grad(v)) * dx
    return msh, t, dt, u, F, J, u_ex


def _assemble_stage_operators(stepper):
    """Assemble the stage Jacobian and preconditioner of a DOLFINx-backed stepper."""
    problem = stepper.solver
    P = problem.P_mat if problem.P_mat is not None else problem.A
    problem.snes.computeJacobian(problem.x, problem.A, P)
    return problem.A, problem.P_mat


@pytest.mark.parametrize("num_stages", [1, 2])
def test_user_jacobian(num_stages):
    """A user-supplied J replaces the automatically derived stage Jacobian."""
    lu = {"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True}

    msh, t, dt, u, F, J, u_ex = _manufactured_heat_problem()
    stepper = StageDerivativeTimeStepper(
        F,
        GaussLegendre(num_stages),
        t,
        dt,
        u,
        solver_parameters={**lu, "petsc_options_prefix": f"auto_J_{num_stages}_"},
        backend="dolfinx",
    )
    A_auto, _ = _assemble_stage_operators(stepper)

    msh, t, dt, u, F, J, u_ex = _manufactured_heat_problem()
    stepper = StageDerivativeTimeStepper(
        F,
        GaussLegendre(num_stages),
        t,
        dt,
        u,
        J=2 * J,
        solver_parameters={**lu, "petsc_options_prefix": f"user_J_{num_stages}_"},
        backend="dolfinx",
    )
    A_user, _ = _assemble_stage_operators(stepper)
    assert np.isclose(A_user.norm(), 2 * A_auto.norm())

    msh, t, dt, u, F, J, u_ex = _manufactured_heat_problem()
    stepper = StageDerivativeTimeStepper(
        F,
        GaussLegendre(num_stages),
        t,
        dt,
        u,
        J=J,
        solver_parameters={**lu, "petsc_options_prefix": f"exact_J_{num_stages}_"},
        backend="dolfinx",
    )
    stepper.advance()
    t.assign(float(t) + float(dt))
    error = norm(u - u_ex, norm_type="L2", mesh=msh)
    assert error < 1e-10, f"Error {error} exceeds tolerance"


@pytest.mark.parametrize("num_stages", [1, 2])
def test_user_preconditioner(num_stages):
    """A user-supplied Jp is assembled into a separate preconditioner matrix."""
    msh, t, dt, u, F, J, u_ex = _manufactured_heat_problem()
    stepper = StageDerivativeTimeStepper(
        F,
        GaussLegendre(num_stages),
        t,
        dt,
        u,
        Jp=2 * J,
        solver_parameters={
            "ksp_type": "gmres",
            "ksp_rtol": 1e-12,
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
            "petsc_options_prefix": f"user_Jp_{num_stages}_",
        },
        backend="dolfinx",
    )
    A, P = _assemble_stage_operators(stepper)
    assert P is not None
    assert np.isclose(P.norm(), 2 * A.norm())

    stepper.advance()
    t.assign(float(t) + float(dt))
    error = norm(u - u_ex, norm_type="L2", mesh=msh)
    assert error < 1e-10, f"Error {error} exceeds tolerance"


@pytest.mark.parametrize("num_stages", [1, 2])
@pytest.mark.parametrize("use_Jp", [False, True])
def test_bilinear_form(num_stages, use_Jp):
    """A bilinear F is solved as a linear stage problem."""
    msh, t, dt, u, F, J, u_ex = _manufactured_heat_problem()
    F = ufl.replace(F, {u: TrialFunction(u.function_space)})
    stepper = StageDerivativeTimeStepper(
        F,
        GaussLegendre(num_stages),
        t,
        dt,
        u,
        Jp=2 * J if use_Jp else None,
        solver_parameters={
            "ksp_type": "gmres",
            "ksp_rtol": 1e-12,
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
            "petsc_options_prefix": f"bilinear_{num_stages}_{use_Jp}_",
        },
        backend="dolfinx",
    )
    assert (stepper.solver.P_mat is not None) == use_Jp

    stepper.advance()
    t.assign(float(t) + float(dt))
    error = norm(u - u_ex, norm_type="L2", mesh=msh)
    assert error < 1e-10, f"Error {error} exceeds tolerance"
    num_steps, _, num_linear_iterations = stepper.solver_stats()
    assert num_steps == 1
    assert num_linear_iterations > 0


@pytest.mark.parametrize("bilinear", [False, True])
def test_form_compiler_parameters(bilinear):
    """form_compiler_parameters reach FFCx as DOLFINx form_compiler_options."""
    _msh, t, dt, u, F, _J, _u_ex = _manufactured_heat_problem()
    if bilinear:
        F = ufl.replace(F, {u: TrialFunction(u.function_space)})

    # FFCx applies the "verbosity" option to its logger on every (cached) compilation
    ffcx_logger = logging.getLogger("ffcx")
    level = ffcx_logger.level
    ffcx_logger.setLevel(logging.WARNING)
    try:
        StageDerivativeTimeStepper(
            F,
            GaussLegendre(1),
            t,
            dt,
            u,
            form_compiler_parameters={"verbosity": logging.ERROR},
            solver_parameters={"petsc_options_prefix": f"fcp_{bilinear}_"},
            backend="dolfinx",
        )
        assert ffcx_logger.level == logging.ERROR
    finally:
        ffcx_logger.setLevel(level)


def test_transpose_nullspace():
    """A transpose nullspace is attached to the stage operator."""
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))

    MC = MeshConstant(msh, backend="dolfinx")
    t = MC.Constant(0.0)
    dt = MC.Constant(0.1)

    u = dolfinx.fem.Function(V)
    v = TestFunction(V)
    # Symmetric stage operator with constants in its kernel and in that of its transpose
    F = inner(grad(Dt(u)), grad(v)) * dx + inner(grad(u), grad(v)) * dx

    constant = dolfinx.fem.Function(V)
    constant.x.array[:] = 1.0
    constant.x.petsc_vec.normalize()
    nullspace = PETSc.NullSpace().create(vectors=[constant.x.petsc_vec], comm=msh.comm)

    stepper = StageDerivativeTimeStepper(
        F,
        GaussLegendre(1),
        t,
        dt,
        u,
        transpose_nullspace=nullspace,
        solver_parameters={"petsc_options_prefix": "transpose_nsp_"},
        backend="dolfinx",
    )
    problem = stepper.solver
    problem.snes.computeJacobian(problem.x, problem.A, problem.A)

    transpose_nullspace = problem.A.getTransposeNullSpace()
    assert transpose_nullspace.handle != 0
    assert len(transpose_nullspace.getVecs()) == 1
    assert transpose_nullspace.test(problem.A.transpose())


@pytest.mark.parametrize("num_stages", [1, 2])
def test_vector_dirichlet_bc(num_stages):
    """A Dirichlet BC on a vector-valued (blocked) space is reconstructed on its stage spaces."""
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))

    MC = MeshConstant(msh, backend="dolfinx")
    t = MC.Constant(0.0)
    dt = MC.Constant(0.1)
    (x, y) = SpatialCoordinate(msh)
    uexact = as_vector([t + x, t + y])

    u = dolfinx.fem.Function(V)
    u.interpolate(dolfinx.fem.Expression(uexact, V.element.interpolation_points))
    v = TestFunction(V)
    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - inner(Dt(uexact), v) * dx

    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        V, msh.topology.dim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology)
    )
    stepper = StageDerivativeTimeStepper(
        F,
        GaussLegendre(num_stages),
        t,
        dt,
        u,
        bcs=[dirichletbc(uexact, boundary_dofs, V)],
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "petsc_options_prefix": f"vector_bc_{num_stages}_",
        },
        backend="dolfinx",
    )
    stepper.advance()
    t.assign(float(t) + float(dt))

    error = norm(u - uexact, norm_type="L2", mesh=msh)
    assert error < 1e-10, f"Error {error} exceeds tolerance"
