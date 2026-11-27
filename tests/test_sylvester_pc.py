import pytest
from firedrake import (
    Constant, DirichletBC, Function, FunctionSpace, RectangleMesh,
    SpatialCoordinate, TestFunction, TestFunctions, UnitSquareMesh,
    VectorFunctionSpace, as_vector, div, dot, dx, errornorm, exp, grad,
    inner, norm, pi, sin, split
)
from firedrake.petsc import PETSc
from irksome import (
    Dt, GaussLegendre, MeshConstant, RadauIIA, TimeStepper
)
from irksome.tools import AI, IA


def lu_params(nonlinear=False):
    return {"snes_type": "newtonls" if nonlinear else "ksponly",
            "snes_linesearch_type": "basic",
            "snes_rtol": 1e-12, "snes_atol": 1e-14,
            "mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"}


def sylvester_params(nonlinear=False, **extra):
    params = {"snes_type": "newtonls" if nonlinear else "ksponly",
              "snes_linesearch_type": "basic",
              "snes_rtol": 1e-12, "snes_atol": 1e-14,
              "mat_type": "matfree",
              "ksp_type": "fgmres", "ksp_rtol": 1e-10,
              "pc_type": "python",
              "pc_python_type": "irksome.IRKSylvesterPC",
              "sylvester_ksp_rtol": 1e-11}
    params.update(extra)
    return params


def get_pc(stepper):
    return stepper.solver.snes.ksp.pc.getPythonContext()


def run(problem, params, tableau, dts, **kwargs):
    """Take one step for each time step in dts and return the solution
    and the outer Krylov iterations per step."""
    F, u, bcs, t, dt = problem()
    stepper = TimeStepper(F, tableau, t, dt, u, bcs=bcs,
                          solver_parameters=params, **kwargs)
    its = []
    for dt_ in dts:
        dt.assign(dt_)
        stepper.advance()
        t.assign(float(t) + dt_)
        its.append(stepper.solver.snes.getLinearSolveIterations()
                   / max(1, stepper.solver.snes.getIterationNumber()))
    return u, its, stepper


def heat(N=8, forcing=True):
    msh = UnitSquareMesh(N, N)

    def problem():
        V = FunctionSpace(msh, "CG", 2)
        MC = MeshConstant(msh)
        t, dt = MC.Constant(0.0), MC.Constant(0.1)
        x, y = SpatialCoordinate(msh)
        uexact = sin(pi * x) * sin(pi * y) * exp(-t)
        u = Function(V).interpolate(uexact)
        v = TestFunction(V)
        F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx
        if forcing:
            F -= inner(Dt(uexact) - div(grad(uexact)), v) * dx
        return F, u, DirichletBC(V, 0, "on_boundary"), t, dt
    return problem


def nonlinear_heat(N=8):
    msh = UnitSquareMesh(N, N)

    def problem():
        V = FunctionSpace(msh, "CG", 2)
        MC = MeshConstant(msh)
        t, dt = MC.Constant(0.0), MC.Constant(0.1)
        x, y = SpatialCoordinate(msh)
        uexact = 2 * sin(pi * x) * sin(pi * y) * (1 + sin(2 * pi * t) / 2)
        u = Function(V).interpolate(uexact)
        v = TestFunction(V)
        F = (inner(Dt(u), v) * dx + inner((1 + u**2) * grad(u), grad(v)) * dx
             - inner(Dt(uexact) - div((1 + uexact**2) * grad(uexact)), v) * dx)
        return F, u, DirichletBC(V, 0, "on_boundary"), t, dt
    return problem


def channel(nu=1, convection=False, N=4):
    """Stokes or Navier-Stokes flow in a channel with a time-dependent
    parabolic inflow and a do-nothing outflow."""
    msh = RectangleMesh(4 * N, N, 4.0, 1.0)

    def problem():
        V = VectorFunctionSpace(msh, "CG", 2)
        W = V * FunctionSpace(msh, "CG", 1)
        MC = MeshConstant(msh)
        t, dt = MC.Constant(0.0), MC.Constant(0.1)
        x, y = SpatialCoordinate(msh)
        up = Function(W)
        u, p = split(up)
        v, q = TestFunctions(W)
        F = (inner(Dt(u), v) * dx + Constant(nu) * inner(grad(u), grad(v)) * dx
             - p * div(v) * dx - div(u) * q * dx)
        if convection:
            F += inner(dot(grad(u), u), v) * dx
        inflow = as_vector([4 * y * (1 - y) * (1 + sin(pi * t) / 2), 0])
        bcs = [DirichletBC(W.sub(0), inflow, 1),
               DirichletBC(W.sub(0), 0, (3, 4))]
        return F, up, bcs, t, dt
    return problem


@pytest.mark.parametrize("stage_type,splitting",
                         [("deriv", AI), ("deriv", IA), ("value", AI)])
@pytest.mark.parametrize("tableau", [RadauIIA(1), RadauIIA(3), GaussLegendre(2)])
def test_heat(stage_type, splitting, tableau):
    """For a linear problem the preconditioner is a solver."""
    dts = [0.1] * 3
    kwargs = dict(stage_type=stage_type, splitting=splitting)
    problem = heat()
    ulu, _, _ = run(problem, lu_params(), tableau, dts, **kwargs)
    u, its, _ = run(problem, sylvester_params(), tableau, dts, **kwargs)
    assert its == [1, 1, 1]
    assert errornorm(ulu, u) < 1e-9 * norm(ulu)


@pytest.mark.parametrize("linearisation", ["mean", "last", "start"])
def test_nonlinear_heat(linearisation):
    dts = [0.1] * 2
    params = sylvester_params(nonlinear=True,
                              sylvester_linearisation=linearisation,
                              sylvester_residual="sylvester",
                              sylvester_ksp_rtol=1e-3)
    problem = nonlinear_heat()
    ulu, _, _ = run(problem, lu_params(True), RadauIIA(2), dts)
    u, its, stepper = run(problem, params, RadauIIA(2), dts)
    assert get_pc(stepper).linearisation == linearisation
    assert max(its) <= 10
    assert errornorm(ulu, u) < 1e-9 * norm(ulu)


@pytest.mark.parametrize("convection", [False, True])
def test_channel(convection):
    """A DAE: the pressure has no time derivative."""
    dts = [0.1] * 2
    problem = channel(nu=0.1 if convection else 1, convection=convection)
    ulu, _, _ = run(problem, lu_params(convection), RadauIIA(2), dts)
    u, its, stepper = run(problem, sylvester_params(convection), RadauIIA(2), dts)
    assert get_pc(stepper).algebraic_fields == [1]
    if not convection:
        assert its == [1, 1]
    assert errornorm(ulu, u) < 1e-8 * norm(ulu)


def test_time_step_change():
    """The preconditioner follows changes of the time step."""
    dts = [0.1, 0.05, 0.2]
    problem = heat()
    ulu, _, _ = run(problem, lu_params(), RadauIIA(2), dts)
    u, its, stepper = run(problem, sylvester_params(), RadauIIA(2), dts)
    assert its == [1, 1, 1]
    assert get_pc(stepper).dt == 0.2
    assert errornorm(ulu, u) < 1e-9 * norm(ulu)

    problem = channel()
    ulu, _, _ = run(problem, lu_params(), RadauIIA(2), dts)
    u, its, stepper = run(problem, sylvester_params(), RadauIIA(2), dts[:1])
    pc = get_pc(stepper)
    beta, state = pc.beta, pc.M.petscmat.stateGet()
    for dt_ in dts[1:]:
        stepper.dt.assign(dt_)
        stepper.advance()
        stepper.t.assign(float(stepper.t) + dt_)
        its.append(stepper.solver.snes.getLinearSolveIterations())
    assert its == [1, 1, 1]
    # beta changes with the time step, but M_1 and its solver do not
    assert pc.beta != beta
    assert pc.M.petscmat.stateGet() == state
    assert errornorm(ulu, u) < 1e-8 * norm(ulu)


def test_low_rank_rhs():
    """Without forcing and with a zero initial guess, the stage residuals
    are equal: the right-hand side has rank one."""
    problem = heat(forcing=False)
    u, its, stepper = run(problem, sylvester_params(), RadauIIA(3), [0.1])
    pc = get_pc(stepper)
    assert pc.rank == 1
    assert pc.ksp.getPythonContext().rank == 1
    ulu, _, _ = run(problem, lu_params(), RadauIIA(3), [0.1])
    assert errornorm(ulu, u) < 1e-9 * norm(ulu)

    # zero right-hand side
    pcobj = stepper.solver.snes.ksp.pc
    x, y = pcobj.getOperators()[0].createVecs()
    x.zeroEntries()
    y.set(1.0)
    pc.apply(pcobj, x, y)
    assert pc.rank == 0
    assert y.norm() == 0


def iterative_inner():
    return {"ksp_type": "gmres", "ksp_rtol": 1e-12, "pc_type": "jacobi"}


@pytest.mark.skipif(not PETSc.Sys.hasExternalPackage("hpddm"),
                    reason="PETSc was built without HPDDM")
def test_fused():
    """Block solves with K and M give the same result."""
    dts = [0.1] * 2
    common = dict(sylvester_K=iterative_inner(), sylvester_M=iterative_inner())
    problem = heat()
    u1, its1, _ = run(problem, sylvester_params(**common), RadauIIA(3), dts)
    u2, its2, _ = run(problem, sylvester_params(sylvester_fuse=True, **common),
                      RadauIIA(3), dts)
    assert its1 == its2
    assert errornorm(u1, u2) < 1e-9 * norm(u1)


@pytest.mark.parametrize("problem", ["heat", "channel"])
def test_parallel(problem):
    """Also run in parallel: mpiexec -n 2 python -m pytest -k test_parallel"""
    problem = {"heat": heat(), "channel": channel()}[problem]
    dts = [0.1] * 2
    ulu, _, _ = run(problem, lu_params(), RadauIIA(2), dts)
    u, its, _ = run(problem, sylvester_params(), RadauIIA(2), dts)
    assert its == [1, 1]
    assert errornorm(ulu, u) < 1e-8 * norm(ulu)
