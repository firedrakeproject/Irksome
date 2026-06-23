import numpy
import pytest
from firedrake import (
    Constant, DirichletBC, Function, FunctionSpace, MixedVectorSpaceBasis,
    PCG64, RandomGenerator, SpatialCoordinate, TestFunction, TrialFunction,
    PeriodicIntervalMesh, UnitSquareMesh, VectorFunctionSpace, VectorSpaceBasis,
    assemble, as_vector, derivative, div, dot, dx, errornorm, exp, grad, inner, split, solve
)
from irksome import (
    Dt, DiscontinuousGalerkinCollocationScheme,
    ContinuousPetrovGalerkinScheme, GalerkinCollocationScheme,
    IRKAuxiliaryOperatorPC, LobattoIIIC,
    MeshConstant, RadauIIA, TimeStepper, TimeQuadratureLabel
)
from irksome.tools import AI, IA
from irksome.labeling import as_form

# FIXME should no longer be needed after https://github.com/firedrakeproject/firedrake/pull/5079
from firedrake import parameters
parameters["default_sub_matrix_type"] = "aij"

# Tests that various PCs are actually getting the right answer.


def Fubc(V, t, uexact):
    u = Function(V)
    u.interpolate(uexact)
    v = TestFunction(V)
    rhs = Dt(uexact) - div(grad(uexact)) - uexact * (1-uexact)
    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(u*(1-u), v)*dx - inner(rhs, v)*dx
    bc = DirichletBC(V, uexact, "on_boundary")
    return (F, u, bc)


class myPC(IRKAuxiliaryOperatorPC):
    def getNewForm(self, pc, u0, test):
        appctx = self.get_appctx(pc)
        bcs = appctx["stepper"].orig_bcs
        F = inner(Dt(u0), test) * dx + inner(grad(u0), grad(test)) * dx
        return F, bcs


def rd(scheme, **kwargs):
    """When a GalerkinCollocationScheme is provided, this preconditions the
    Jacobian by applying the Rana triangular approximation to the corresponding
    collocation IRK.
    """
    N = 4
    msh = UnitSquareMesh(N, N)

    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)

    deg = 2
    V = FunctionSpace(msh, "CG", deg)

    x, y = SpatialCoordinate(msh)

    uexact = t*(x+y)

    sols = []

    luparams = {"mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu"}

    per_field = {"ksp_type": "preonly",
                 "pc_type": "gamg"}

    ranaLD = {
        "mat_type": "matfree",
        "ksp_type": "gmres",
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "irksome.RanaLD",
        "aux": {
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "multiplicative",
            "fieldsplit": per_field,
        }
    }

    ranaDU = {
        "mat_type": "matfree",
        "ksp_type": "gmres",
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "irksome.RanaDU",
        "aux": {
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "multiplicative",
            "fieldsplit": per_field,
        }
    }

    mypc_params = {
        "mat_type": "matfree",
        "ksp_type": "gmres",
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "test_pc.myPC",
        "aux": {
            "pc_type": "lu",
        }
    }

    params = [luparams, ranaLD, ranaDU, mypc_params]

    for solver_parameters in params:
        F, u, bc = Fubc(V, t, uexact)

        stepper = TimeStepper(F, scheme, t, dt, u, bcs=bc,
                              solver_parameters=solver_parameters, **kwargs)
        stepper.advance()
        sols.append(u)

        num_stages = stepper.num_stages
        for i in range(num_stages):
            ranaDU[f"aux_pc_fieldsplit_{i}_fields"] = num_stages-1-i

    errs = [errornorm(sols[0], uu) for uu in sols[1:]]
    return numpy.max(errs)


@pytest.mark.parametrize("stage_type", ("deriv", "value"))
@pytest.mark.parametrize('butcher_tableau,order', [
    (RadauIIA, 2),
    (LobattoIIIC, 3),
])
def test_pc_acc(butcher_tableau, order, stage_type):
    splitting = {"deriv": AI, "value": IA}[stage_type]
    assert rd(butcher_tableau(order), stage_type=stage_type, splitting=splitting) < 1.e-6


@pytest.mark.parametrize("stage_type", ("deriv", "value"))
@pytest.mark.parametrize('quad_scheme,order', [
    ("radau", 2),
])
def test_pc_cpg_collocation(quad_scheme, order, stage_type):
    scheme = GalerkinCollocationScheme(order, quadrature_scheme=quad_scheme, quadrature_degree="auto", stage_type=stage_type)
    assert rd(scheme) < 1.e-6


@pytest.mark.parametrize("deriv_type", ("strong", "weak"))
@pytest.mark.parametrize('order', [2])
def test_pc_dg_collocation(order, deriv_type):
    scheme = DiscontinuousGalerkinCollocationScheme(order, quadrature_degree="auto", deriv_type=deriv_type)
    assert rd(scheme) < 1.e-6


@pytest.mark.parametrize('scheme', [
    DiscontinuousGalerkinCollocationScheme(0, quadrature_scheme="radau"),
    DiscontinuousGalerkinCollocationScheme(1, quadrature_scheme="radau"),
    GalerkinCollocationScheme(1, quadrature_scheme="radau", stage_type="deriv"),
    GalerkinCollocationScheme(1, quadrature_scheme="radau", stage_type="value"),
    GalerkinCollocationScheme(2, quadrature_scheme="radau", stage_type="deriv"),
    GalerkinCollocationScheme(2, quadrature_scheme="radau", stage_type="value"),
], ids=repr)
def test_git_irk_equivalence(scheme):
    N = 5
    msh = UnitSquareMesh(N, N)
    V = VectorFunctionSpace(msh, "CG", 2)
    Q = FunctionSpace(msh, "CG", 1)
    Z = V * Q

    t = Constant(0.0)
    dt = Constant(1.0/N)
    (x, y) = SpatialCoordinate(msh)

    uexact = as_vector([x*t + y**2, -y*t+t*(x**2)])
    pexact = x + y * t

    nu = Constant(0.1)
    u_rhs = Dt(uexact) - div(nu*grad(uexact)) + dot(grad(uexact), uexact) + grad(pexact)
    p_rhs = -div(uexact)

    z = Function(Z)
    ztest = TestFunction(Z)
    u, p = split(z)
    v, q = split(ztest)

    F = (inner(Dt(u), v)*dx
         + inner(nu*grad(u), grad(v))*dx
         + inner(dot(grad(u), u), v)*dx
         - inner(p, div(v))*dx
         - inner(div(u), q)*dx
         - inner(u_rhs, v)*dx
         - inner(p_rhs, q)*dx)

    bcs = [DirichletBC(Z.sub(0), uexact, "on_boundary")]
    nsp = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=msh.comm)])
    pconst = Function(Q).interpolate(Constant(1))
    vol = assemble(pconst*dx)

    u, p = z.subfunctions
    u.interpolate(uexact)
    p.interpolate(pexact)

    sparams = {
        "mat_type": "matfree",
        "ksp_type": "gmres",
        "ksp_view_eigenvalues": None,
        "ksp_converged_reason": None,
        "snes_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "irksome.IRKAuxiliaryOperatorPC",
        "aux": {
            "mat_type": "nest",
            "sub_mat_type": "aij",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    }

    kwargs = {}
    if isinstance(scheme, ContinuousPetrovGalerkinScheme):
        kwargs["aux_indices"] = [1]
    stepper = TimeStepper(F, scheme, t, dt, z,
                          bcs=bcs, solver_parameters=sparams,
                          nullspace=nsp,
                          pre_apply_bcs=False,
                          **kwargs)

    for step in range(N):
        stepper.advance()
        pavg = (1/vol) * assemble(p*dx)
        p.assign(p - pavg * pconst)
        assert numpy.allclose(stepper.solver.snes.ksp.computeEigenvalues(), 1.0)
        t.assign(t + dt)


@pytest.mark.parametrize('order', (1, 2, 3))
@pytest.mark.parametrize('family', (GalerkinCollocationScheme, RadauIIA))
def test_stokes_augmented_lagrangian_preconditioner(family, order):
    N = 5
    msh = UnitSquareMesh(N, N)
    V = VectorFunctionSpace(msh, "CG", 2, variant="alfeld")
    Q = FunctionSpace(msh, "DG", 1, variant="integral,alfeld")
    Z = V * Q

    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(1.0/N)
    (x, y) = SpatialCoordinate(msh)

    uexact = as_vector([x*t + y**2, -y*t+t*(x**2)])
    pexact = x + y * t

    u_rhs = Dt(uexact) - div(grad(uexact)) + grad(pexact)
    p_rhs = -div(uexact)

    z = Function(Z)

    # Use a TrialFunction to test the LinearVariationalSolver interface
    ztrial = TrialFunction(Z)
    ztest = TestFunction(Z)
    u, p = split(ztrial)
    v, q = split(ztest)

    scheme = family(order)
    if isinstance(scheme, ContinuousPetrovGalerkinScheme):
        scheme = family(order, stage_type="value")
        Llow = TimeQuadratureLabel(2*order-2, scheme="radau")
    else:
        Llow = lambda x: x
    F = (
        inner(grad(u), grad(v))*dx
        + Llow(
            inner(Dt(u), v)*dx
            - inner(p, div(v))*dx
            - inner(div(u), q)*dx
        )
        - inner(u_rhs, v)*dx
        - inner(p_rhs, q)*dx
    )
    # Augmented Lagrangian preconditioner
    Jp = inner(grad(u), grad(v))*dx + Llow(inner(Dt(u), v)*dx + inner(div(u), div(v))*dx + inner(p, q)*dx)

    bcs = [DirichletBC(Z.sub(0), uexact, "on_boundary")]

    sparams = {
        "mat_type": "matfree",
        "pmat_type": "aij",
        "ksp_max_it": 15,
        "ksp_rtol": 1E-5,
        "ksp_view_eigenvalues": None,
        "ksp_converged_reason": None,
        "pc_factor_mat_solver_type": "mumps",
    }
    if order == 1:
        sparams["ksp_type"] = "minres"
        sparams["ksp_norm_type"] = "preconditioned"
        sparams["pc_type"] = "cholesky"
    else:
        sparams["ksp_type"] = "gmres"
        sparams["ksp_pc_side"] = "right"
        sparams["pc_type"] = "lu"

    stepper = TimeStepper(F, scheme, t, dt, z,
                          bcs=bcs, Jp=Jp,
                          solver_parameters=sparams,
                          constant_jacobian=True)

    ksp = stepper.solver.snes.ksp
    u, p = z.subfunctions
    u.interpolate(uexact)
    p.interpolate(pexact)
    for step in range(N):
        stepper.stages.zero()
        stepper.advance()
        t.assign(float(t) + float(dt))

        eigs = ksp.computeEigenvalues()

        # There should be a single positive eigenvalue equal to one
        assert numpy.allclose(eigs[numpy.real(eigs) > 0], 1)

        # The negative eigenvalues should be bounded from the left
        if isinstance(scheme, ContinuousPetrovGalerkinScheme):
            bound = 2/3
        else:
            bound = 1/2
        assert min(numpy.real(eigs)) >= -bound

        # Eigenvalues must be bounded away from 0
        assert abs(min(eigs, key=abs)) > 0.05

        # All eigenvalues must be real
        assert numpy.allclose(numpy.imag(eigs), 0)


@pytest.mark.parametrize('stage_type', ("value", "deriv"))
@pytest.mark.parametrize('order', (1, 2))
def test_bbm_underintegrated_preconditioner(stage_type, order):
    N = 8000
    L = 100
    msh = PeriodicIntervalMesh(N, L)
    x, = SpatialCoordinate(msh)

    t = Constant(0)
    inv_dt = N // (10 * L)
    tfinal = 18
    Nt = tfinal * inv_dt
    dt = Constant(tfinal / Nt)
    Nt = 10

    c = Constant(0.5)
    center = Constant(40.0)
    delta = -c * center

    def sech(x):
        return 2 / (exp(x) + exp(-x))

    uexact = 3 * c**2 / (1-c**2) * sech(0.5 * (c * x - c * t / (1 - c ** 2) + delta))**2

    def h1inner(u, v):
        return inner(u, v) + inner(grad(u), grad(v))

    def I1(u):
        return u * dx

    def I2(u):
        return 0.5 * h1inner(u, u) * dx

    def I3(u):
        return (u**2 / 2 + u**3 / 6) * dx

    V = FunctionSpace(msh, "Hermite", 3)
    u = Function(V)
    v = TestFunction(V)
    w = TrialFunction(V)

    a = h1inner(w, v) * dx
    solve(a == h1inner(uexact, v)*dx, u)

    time_order_high = 3 * order - 1
    Lhigh = TimeQuadratureLabel(time_order_high)

    dHdu = derivative(I3(u), u, v)
    F = h1inner(Dt(u), v)*dx + Lhigh(-dHdu(v.dx(0)))
    scheme = GalerkinCollocationScheme(order, stage_type=stage_type)

    # Drop quadrature labels to define an under-integrated preconditioner
    Jp = as_form(F)

    sparams = {
        "snes_monitor": None,
        "mat_type": "matfree",
        "pmat_type": "aij",
        "ksp_type": "gmres",
        "ksp_max_it": 5,
        "snes_atol": 0,
        "snes_rtol": 1E-14,
        "ksp_atol": 0,
        "ksp_rtol": 1E-12,
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    stepper = TimeStepper(F, scheme, t, dt, u,
                          Jp=Jp,
                          solver_parameters=sparams)

    times = [float(t)]
    functionals = (I1(u), I2(u), I3(u))
    invariants = [tuple(map(assemble, functionals))]
    for _ in range(Nt):
        stepper.advance()

        invariants.append(tuple(map(assemble, functionals)))
        i1, i2, i3 = invariants[-1]
        t.assign(float(t) + float(dt))
        times.append(float(t))

        print(f'{float(t):.15f}, {i1:.15f}, {i2:.15f}, {i3:.15f}')

        iprev = invariants[-2]
        icur = invariants[-1]
        assert numpy.allclose(iprev[0], icur[0], atol=1e-14)
        assert numpy.allclose(iprev[2], icur[2], atol=1e-14)


@pytest.mark.parametrize("scheme", [
    DiscontinuousGalerkinCollocationScheme(1, quadrature_degree="auto"),
    GalerkinCollocationScheme(2, stage_type="deriv", quadrature_degree="auto"),
    GalerkinCollocationScheme(2, stage_type="value", quadrature_degree="auto"),
], ids=repr)
def test_git_irk_equivalence_scheme_Jp(scheme):
    # discretization
    N = 32
    msh = UnitSquareMesh(N, N)
    V = FunctionSpace(msh, "CG", 1)

    t = Constant(0.0)
    dt = Constant(1.e-1)
    Tfinal = 0.5
    Nt = int(Tfinal / float(dt))

    # seed IC once
    rg = RandomGenerator(PCG64(seed=0))
    uic = rg.uniform(V)

    u = Function(V)
    v = TestFunction(V)
    eps = Constant(1.e-2)
    F = (inner(Dt(u), v) * dx
         + eps**2 * inner(grad(u), grad(v)) * dx
         + inner(u**3 - u, v) * dx)

    def run(solver_parameters, **kwargs):
        t.assign(0)
        u.assign(uic)
        stepper = TimeStepper(F, scheme, t, dt, u,
                              solver_parameters=solver_parameters, **kwargs)
        for _ in range(Nt):
            stepper.advance()
            t.assign(t + dt)

        nsteps, snes_its, ksp_its = stepper.solver_stats()
        return snes_its, ksp_its

    # Blunt Newton with direct solve, full Jacobian
    solver_parameters = {
        "snes_atol": 1e-12,
        "snes_rtol": 1e-10,
        "snes_linesearch_type": "bisection",
        "snes_divergence_tolerance": 1e12,
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
    base_snes_its, base_ksp_its = run(solver_parameters)
    ubase = Function(u)

    # Newton-Krylov with the Jacobian preconditioned by an IRK collocation scheme, accessed by IRK Auxiliary Operator
    solver_parameters = {
        # Outer (nonlinear) solver
        "snes_monitor": None,
        "snes_atol": 1e-12,
        "snes_rtol": 1e-10,
        "snes_linesearch_type": "bisection",
        "snes_divergence_tolerance": 1e12,
        "ksp_converged_reason": None,
        "ksp_view_eigenvalues": None,
        "ksp_type": "gmres",
        "ksp_rtol": 1.e-12,
        "ksp_atol": 1.e-14,
        "pc_type": "python",
        "pc_python_type": "irksome.IRKAuxiliaryOperatorPC",
        "aux_pc_type": "lu",
    }
    irkaux_snes_its, irkaux_ksp_its = run(solver_parameters)
    assert errornorm(ubase, u) < 1E-12
    assert base_snes_its == irkaux_snes_its

    # Newton-Krylov with Jacobian preconditioned by (under-integrated) cPG collocation, through the front door.
    solver_parameters = {
        "snes_monitor": None,
        "snes_atol": 1e-12,
        "snes_rtol": 1e-10,
        "snes_linesearch_type": "bisection",
        "snes_divergence_tolerance": 1e12,
        "ksp_converged_reason": None,
        "ksp_view_eigenvalues": None,
        "ksp_type": "gmres",
        "ksp_rtol": 1.e-12,
        "ksp_atol": 1.e-14,
        "pc_type": "lu",
    }
    scheme_Jp = scheme.as_collocation_scheme()
    aux_snes_its, aux_ksp_its = run(solver_parameters, scheme_Jp=scheme_Jp)
    assert errornorm(ubase, u) < 1E-12
    assert base_snes_its == aux_snes_its
    assert irkaux_ksp_its == aux_ksp_its
