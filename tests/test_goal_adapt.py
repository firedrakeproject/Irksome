"""Goal-oriented adaptivity in TIME for the heat equation with a pulse in time.

The spatial mesh is fine and fixed; the time grid starts coarse (4 slabs) and DWR
bisects the slabs that matter for the goal J(u) = int_0^T int_Omega u dx dt.
"""
import pytest
import numpy
from firedrake import *
from irksome import DiscontinuousGalerkinScheme, Dt, TimeStepper

from irksome.goal_adapt import dwr_spacetime_adaptivity, print_estimates, reference_goal, solve_primal

T = 1.0
t = Constant(0.0)
dt = Constant(0.0)


# def heat(mesh):
#     """Build the problem on the given mesh."""
#     V = FunctionSpace(mesh, "CG", 2)
#     u, v = Function(V), TestFunction(V)
#     x, y = SpatialCoordinate(mesh)
#     dxq = dx(degree=6)
#     pulse = exp(-((t - 0.6) / 0.05) ** 2)
#     f = 50 * pulse * sin(pi * x) * sin(pi * y)
#     F = inner(Dt(u), v) * dxq + inner(grad(u), grad(v)) * dxq - inner(f, v) * dxq
#     bcs = DirichletBC(V, 0, "on_boundary")
#     u.interpolate(sin(pi * x) * sin(pi * y))
#     goal = u * dxq
#     return F, u, bcs, goal


def heat_1d(mesh):
    """u_t - u_xx = f on (0, 1), u = 0 on the boundary, goal J(u) = int_0^T int u dx dt."""
    V = FunctionSpace(mesh, "CG", 2)
    u = Function(V)
    v = TestFunction(V)
    x, = SpatialCoordinate(mesh)
    f = sin(pi * x)

    F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - inner(f, v) * dx
    bcs = DirichletBC(V, 0, "on_boundary")
    u.interpolate(sin(pi * x))
    goal = u * dx
    return F, u, bcs, goal


@pytest.mark.parametrize("order", [0])
@pytest.mark.parametrize("problem", [heat_1d])
def test_goal_adapt(order, problem):
    mesh = UnitIntervalMesh(8)
    nodes = numpy.linspace(0.0, T, 4 + 1)
    scheme = DiscontinuousGalerkinScheme(order, quadrature_degree="auto")

    J_ref = reference_goal(problem, mesh, T, t, dt, nsteps=256, order=2)
    print(f"reference J = {J_ref:.10e}", flush=True)

    eta = numpy.inf
    estimates = []                                   # record of each DWR round: slab count, J_h, error, eta, Ieff...
    # while abs(eta) > tol and len(estimates) < 8:
    for _ in range(5):
        F, u, bcs, goal = problem(mesh)
        stepper = TimeStepper(F, scheme, t, dt, u, bcs=bcs,
                              solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        primal = solve_primal(stepper, nodes)        # one record (u_in, stages) per slab
        nodes, eta = dwr_spacetime_adaptivity(stepper, nodes, primal, goal,
                                              exact_goal=J_ref, estimates=estimates)
        row = estimates[-1]
        # print(f"iteration {row['iteration']}: {row['Nt']} slabs, eta = {eta:.3e}, "
        #    f"error = {row['error']:.3e}, Ieff = {row['Ieff']:.3f}", flush=True)
        # assert Ieff > 0.95
        assert abs(row["Ieff"] - 1) < 0.05
        # rate = (log(error_prev) - log(error)) / (log(Nt) - log(Nt_prev))
        # assert rate > convrate

        if len(estimates) > 1:
            prev = estimates[-2]
            rate = ((numpy.log(abs(prev["error"])) - numpy.log(abs(row["error"])))
                    / (numpy.log(row["Nt"]) - numpy.log(prev["Nt"])))
            print(f"rate = {rate:.2f}")
            assert rate > order + 0.9

    print()
    print_estimates(estimates)
    print("final time nodes:", numpy.round(nodes, 4))
