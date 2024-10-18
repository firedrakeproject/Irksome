import pytest
from firedrake import *
from irksome import Dt, LobattoIIIC, MeshConstant, RadauIIA, TimeStepper
from irksome.tools import AI, IA
from ufl.algorithms import expand_derivatives

# test the accuracy of the 2d Stokes heat equation using CG elements
# and LobattoIIIC time integration.  Particular issue being tested is
# enforcement of the pressure BC, which is a field that doesn't have
# a time derivative on it.


def StokesTest(N, butcher_tableau, stage_type="deriv", splitting=AI):
    mesh0 = UnitSquareMesh(N//4, N//4)
    mh = MeshHierarchy(mesh0, 2)
    mesh = mh[-1]

    ns = butcher_tableau.num_stages

    Ve = VectorElement("CG", mesh.ufl_cell(), 2)
    Pe = FiniteElement("CG", mesh.ufl_cell(), 1)
    Ze = MixedElement([Ve, Pe])
    Z = FunctionSpace(mesh, Ze)

    MC = MeshConstant(mesh)
    t = MC.Constant(0.0)
    dt = MC.Constant(1.0/N)
    (x, y) = SpatialCoordinate(mesh)

    uexact = as_vector([x*t + y**2, -y*t+t*(x**2)])
    pexact = x + y * t

    u_rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact)) + grad(pexact)
    p_rhs = -div(uexact)

    z = Function(Z)
    test_z = TestFunction(Z)
    (u, p) = split(z)
    (v, q) = split(test_z)
    F = (inner(Dt(u), v)*dx
         + inner(grad(u), grad(v))*dx
         - inner(p, div(v))*dx
         - inner(q, div(u))*dx
         - inner(u_rhs, v)*dx
         - inner(p_rhs, q)*dx)

    bcs = [DirichletBC(Z.sub(0), uexact, "on_boundary")]
    nsp = [(1, VectorSpaceBasis(constant=True))]

    u, p = z.subfunctions
    u.interpolate(uexact)
    p.interpolate(pexact-assemble(pexact*dx))

    ns = butcher_tableau.num_stages
    ind_pressure = ",".join([str(2*i+1) for i in range(ns)])
    solver_params = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "fgmres",
        "ksp_max_it": 200,
        "ksp_gmres_restart": 30,
        "ksp_rtol": 1.e-8,
        "ksp_atol": 1.e-13,
        "pc_type": "mg",
        "pc_mg_type": "multiplicative",
        "pc_mg_cycles": "v",
        "mg_levels": {
            "ksp_type": "chebyshev",
            "ksp_max_it": 3,
            "ksp_convergence_test": "skip",
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMVankaPC",
            "pc_vanka_construct_dim": 0,
            "pc_vanka_sub_sub_pc_type": "lu",
            "pc_vanka_sub_sub_pc_factor_mat_solver_type": "umfpack",
            "pc_vanka_exclude_subspaces": ind_pressure},
        "mg_coarse": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200}
    }

    stepper = TimeStepper(F, butcher_tableau, t, dt, z,
                          stage_type=stage_type,
                          bcs=bcs, solver_parameters=solver_params,
                          nullspace=nsp)

    while (float(t) < 1.0):
        if (float(t) + float(dt) > 1.0):
            dt.assign(1.0 - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    (u, p) = z.subfunctions

    # check error mod the constants
    perr = norm(pexact - p - assemble((pexact-p) * dx))

    print(perr, errornorm(uexact, u))
    return errornorm(uexact, u) + perr


# make sure things run on a driven cavity.  We time step a while
# and check that the velocity is the right size (as observed from
# a "by-hand" backward Euler code in Firedrake
def NSETest(butch, stage_type, splitting):
    N = 4
    M = UnitSquareMesh(N, N)
    mh = MeshHierarchy(M, 2)
    M = mh[-1]

    V = VectorFunctionSpace(M, "CG", 2)
    W = FunctionSpace(M, "CG", 1)
    Z = V * W

    up = Function(Z)
    u, p = split(up)
    v, q = TestFunctions(Z)

    Re = Constant(100.0)

    F = (inner(Dt(u), v) * dx
         + 1.0 / Re * inner(grad(u), grad(v)) * dx
         + inner(dot(grad(u), u), v) * dx
         - p * div(v) * dx
         + div(u) * q * dx
         )

    bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
           DirichletBC(Z.sub(0), 0, (1, 2, 3))]

    nullspace = [(1, VectorSpaceBasis(constant=True))]

    ns = butch.num_stages
    ind_pressure = ",".join([str(2*i+1) for i in range(ns)])
    solver_params = {
        "mat_type": "aij",
        "snes_type": "newtonls",
        "snes_converged_reason": None,
        "snes_linesearch_type": "l2",
        "snes_rtol": 1.e-8,
        "ksp_rtol": 1.e-10,
        "ksp_atol": 1.e-13,
        "ksp_type": "fgmres",
        "ksp_monitor": None,
        "ksp_max_it": 200,
        "ksp_gmres_restart": 30,
        "pc_type": "mg",
        "pc_mg_type": "multiplicative",
        "pc_mg_cycles": "v",
        "mg_levels": {
            "ksp_type": "chebyshev",
            "ksp_max_it": 3,
            "ksp_convergence_test": "skip",
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMVankaPC",
            "pc_vanka_construct_dim": 0,
            "pc_vanka_sub_sub_pc_type": "lu",
            "pc_vanka_sub_sub_pc_factor_mat_solver_type": "umfpack",
            "pc_vanka_exclude_subspaces": ind_pressure},
        "mg_coarse": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200}
    }

    MC = MeshConstant(M)
    t = MC.Constant(0.0)
    dt = MC.Constant(1.0/N)
    stepper = TimeStepper(F, butch,
                          t, dt, up,
                          bcs=bcs,
                          stage_type="value",
                          solver_parameters=solver_params,
                          nullspace=nullspace)

    tfinal = 1.0
    u, p = up.subfunctions
    while (float(t) < tfinal):
        if (float(t) + float(dt) > tfinal):
            dt.assign(1.0 - float(t))
        stepper.advance()
        p -= assemble(p*dx)
        print(float(t), norm(u), norm(p), assemble(p*dx))
        t.assign(float(t) + float(dt))

    return norm(u)


@pytest.mark.parametrize('stage_type', ("deriv", "value"))
@pytest.mark.parametrize('splitting', (AI, IA))
@pytest.mark.parametrize('N', [2**j for j in range(3, 4)])
@pytest.mark.parametrize('time_stages', (2, 3))
@pytest.mark.parametrize('butch', (LobattoIIIC, RadauIIA))
def test_Stokes(N, butch, time_stages, stage_type, splitting):
    error = StokesTest(N, butch(time_stages), stage_type, splitting)
    assert abs(error) < 3e-8


@pytest.mark.parametrize('stage_type', ("deriv", "value"))
@pytest.mark.parametrize('time_stages', (2,))
@pytest.mark.parametrize('butch', (LobattoIIIC, RadauIIA))
def test_NSE(butch, time_stages, stage_type):
    unrm = NSETest(butch(time_stages), stage_type, IA)
    assert abs(unrm - 0.216) < 5e-3


if __name__ == "__main__":
    # test_Stokes(4, RadauIIA, 2, "deriv", IA)
    test_NSE(RadauIIA, 2, "deriv")
