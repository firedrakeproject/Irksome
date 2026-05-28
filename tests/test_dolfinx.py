import pytest


from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl

from ufl import div, grad, inner, dx, as_vector, split, SpatialCoordinate, TestFunctions

from irksome import GaussLegendre, Dt, MeshConstant
from irksome.backends.dolfinx import dirichletbc, norm
from irksome.stage_derivative import StageDerivativeTimeStepper
from irksome.tools import AI

dolfinx = pytest.importorskip("dolfinx")


@pytest.mark.parametrize("num_stages", [1, 2, 3])
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
    dt = MC.Constant(1.0 / N)
    (x, y) = SpatialCoordinate(msh)

    Q, Q_to_W = W.sub(1).collapse()
    nsp = dolfinx.fem.Function(W)
    nsp.x.array[Q_to_W] = 1
    nullspace = [nsp.x]
    dolfinx.la.orthonormalize(nullspace)
    nsp = PETSc.NullSpace().create(
        vectors=[n.petsc_vec for n in nullspace], comm=msh.comm
    )

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

    # Dirichlet BCs (irksome style with UFL expression)
    bc = dirichletbc(uexact, boundary_dofs, W.sub(0))
    bcs = [bc]

    # Initial conditions
    bc_expr = dolfinx.fem.Expression(uexact, W.sub(0).element.interpolation_points)
    z.sub(0).interpolate(bc_expr)
    p_avg = msh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(pexact * dx)), op=MPI.SUM
    )
    z.sub(1).interpolate(
        dolfinx.fem.Expression(pexact - p_avg, W.sub(1).element.interpolation_points)
    )

    solver_parameters = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        # "snes_monitor": None,
        "snes_error_if_not_converged": True,
        # "ksp_monitor": None,
        "snes_max_iter": 50,
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
        "snes_linesearch_type": "none",
        "mat_mumps_icntl_24": 1,
        "mat_mumps_icntl_25": 0,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "petsc_options_prefix": f"IrkSomeStokesSolver{num_stages}",
    }
    print(
        f"Rank {MPI.COMM_WORLD.rank}: {solver_parameters['petsc_options_prefix']} {num_stages=}",
        flush=True,
    )
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
        # nullspace=nsp,
        backend="dolfinx",
    )

    z0 = z.sub(0).collapse()

    V, vel_to_mixed = W.sub(0).collapse()

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
        assert error < 1e-10, f"Error {error} exceeds tolerance at timestep {float(t)}"
