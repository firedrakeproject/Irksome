"""A user-supplied bilinear Jacobian must be evaluated at the stage state.

The stages are perturbed before assembling: at the initial guess the stage
state equals u0 and the two Jacobians coincide.
"""
import numpy as np
import pytest
import ufl
from firedrake import (Function, FunctionSpace, SpatialCoordinate, TestFunction,
                       TrialFunction, UnitIntervalMesh, assemble, cos, dx,
                       errornorm, grad, inner, pi)
from irksome import (Alexander, BDF, ContinuousPetrovGalerkinScheme,
                     DIRKNystromTimeStepper, DiscontinuousGalerkinScheme, Dt,
                     GalerkinCollocationScheme, GaussLegendre, MeshConstant,
                     QinZhang, RadauIIA, StageDerivativeNystromTimeStepper,
                     TimeStepper, lag)
from irksome.dirk_stepper import DIRKTimeStepper
from irksome.multistep import MultistepTimeStepper


# quadrature_degree is pinned so that the residual and the Jacobian are
# integrated with the same rule
schemes = [
    pytest.param(RadauIIA(2), {"stage_type": "deriv"}, id="deriv-RadauIIA2"),
    pytest.param(RadauIIA(2), {"stage_type": "value"}, id="value-RadauIIA2"),
    pytest.param(Alexander(), {"stage_type": "dirk"}, id="dirk-Alexander"),
    pytest.param(ContinuousPetrovGalerkinScheme(2, quadrature_degree=6), {}, id="cpg-CPG(2)"),
    pytest.param(DiscontinuousGalerkinScheme(1, quadrature_degree=6), {}, id="dg-DG(1)"),
    pytest.param(GalerkinCollocationScheme(2, stage_type="deriv", quadrature_degree=6),
                 {"scheme_J": RadauIIA(2)}, id="gcs-deriv-RadauIIA2"),
    pytest.param(BDF(2), {}, id="multistep-BDF(2)"),
]

nystrom_schemes = [
    pytest.param(GaussLegendre(2), StageDerivativeNystromTimeStepper,
                 id="nystrom-GaussLegendre2"),
    pytest.param(QinZhang(), DIRKNystromTimeStepper, id="nystrom-dirk-QinZhang"),
]

# A second step size separates a missing or spurious factor of dt.  The Nystrom
# state depends on the stages through dt**2, so its second point is above one.
dt_values = [1.0, 0.37]
nystrom_dt_values = [1.0, 2.5]


def heat_reaction_problem(dt_value):
    """Heat equation with a cubic reaction, plus its exact bilinear Jacobian.

    The reaction is genuinely nonlinear, so dF/du depends on the state it is
    evaluated at, and that dependence is what the test measures.
    """
    msh = UnitIntervalMesh(4)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(dt_value)
    (x,) = SpatialCoordinate(msh)

    u = Function(V).interpolate(1.0 + 0.5 * cos(pi * x))
    v = TestFunction(V)
    du = TrialFunction(V)

    # pinned so that degree estimation cannot differ between F and Jhand
    dxq = dx(degree=6)

    F = (inner(Dt(u), v) * dxq
         + inner(grad(u), grad(v)) * dxq
         + inner(u ** 3, v) * dxq)

    Jhand = (inner(Dt(du), v) * dxq
             + inner(grad(du), grad(v)) * dxq
             + inner(3 * u ** 2 * du, v) * dxq)

    return F, Jhand, t, dt, u, x


def wave_reaction_problem(dt_value):
    """Second-order wave equation with a cubic reaction, plus its Jacobian.

    The Nystrom state at stage i is u0 + ut0*(c[i]*dt) + Abar[i]*k*dt**2, so a
    nonzero ut0 also separates it from u0 at the initial guess.
    """
    msh = UnitIntervalMesh(4)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(dt_value)
    (x,) = SpatialCoordinate(msh)

    u = Function(V).interpolate(1.0 + 0.5 * cos(pi * x))
    ut = Function(V).interpolate(0.25 * cos(2 * pi * x))
    v = TestFunction(V)
    du = TrialFunction(V)

    dxq = dx(degree=6)

    F = (inner(Dt(u, 2), v) * dxq
         + inner(grad(u), grad(v)) * dxq
         + inner(u ** 3, v) * dxq)

    Jhand = (inner(Dt(du, 2), v) * dxq
             + inner(grad(du), grad(v)) * dxq
             + inner(3 * u ** 2 * du, v) * dxq)

    return F, Jhand, t, dt, u, ut, x


def stage_unknown(stepper, method):
    """The Function the stage variational problem solves for.

    A DIRK has no stage vector: the unknown is the single stage derivative k,
    and the state is ``g + k*(a*dt)``.  Put the accumulators in the
    configuration ``advance`` uses for the first stage.  A multistep method
    solves directly for the new step, which is ``u0`` itself.
    """
    if isinstance(stepper, DIRKTimeStepper):
        k, g, a, c = stepper.kgac
        assert float(method.A[0, 0]) != 0.0
        g.assign(stepper.u0)
        a.assign(float(method.A[0, 0]))
        c.assign(float(method.c[0]))
        return k
    if isinstance(stepper, MultistepTimeStepper):
        return stepper.us[-1]
    return stepper.stages


def nystrom_stage_unknown(stepper):
    """As ``stage_unknown``, for the two Nystrom front ends.

    The DIRK-Nystrom state is ``g1 + k*(abar*dt**2)``, with ``g1`` and ``g2``
    holding the first-stage accumulators that ``advance`` builds.
    """
    if isinstance(stepper, DIRKNystromTimeStepper):
        k, g1, g2, a, abar, c = stepper.kgac
        assert float(stepper.AAbar[0, 0]) != 0.0
        g1.assign(stepper.u0 + stepper.ut0 * (stepper.CC[0] * stepper.dt))
        g2.assign(stepper.ut0)
        a.assign(float(stepper.AA[0, 0]))
        abar.assign(float(stepper.AAbar[0, 0]))
        c.assign(float(stepper.CC[0]))
        return k
    return stepper.stages


def perturb(w, x):
    """Deterministic, nonzero, stage-dependent values for the unknown."""
    for i, wi in enumerate(w.subfunctions):
        wi.interpolate(0.6 * cos((i + 1) * pi * x) - 0.2)


def petsc_mat(form):
    return assemble(form, mat_type="aij").petscmat.copy()


def reldiff(A, B):
    D = A.copy()
    D.axpy(-1.0, B)
    return D.norm() / A.norm()


def compare(J_auto, J_hand, w_auto, w_hand, x):
    """Move the stages and report how far the two Jacobians travelled apart."""
    A_zero = petsc_mat(J_auto)

    perturb(w_auto, x)
    perturb(w_hand, x)

    A_auto = petsc_mat(J_auto)
    A_hand = petsc_mat(J_hand)

    moved = reldiff(A_auto, A_zero)
    diff = reldiff(A_auto, A_hand)
    print(f"stage sensitivity {moved:.3e}, hand-vs-auto {diff:.3e}")
    return moved, diff


@pytest.mark.parametrize("dt_value", dt_values)
@pytest.mark.parametrize("method,kwargs", schemes)
def test_bilinear_jacobian_uses_stage_state(method, kwargs, dt_value):
    F, Jhand, t, dt, u, x = heat_reaction_problem(dt_value)

    auto = TimeStepper(F, method, t, dt, u, **kwargs)
    hand = TimeStepper(F, method, t, dt, u, J=Jhand, **kwargs)

    moved, diff = compare(auto.problem.J, hand.problem.J,
                          stage_unknown(auto, method),
                          stage_unknown(hand, method), x)

    # The perturbation must have teeth: if the correct Jacobian does not move,
    # the assertion below would pass on the frozen form as well.
    assert moved > 1e-3
    assert diff < 1e-12


@pytest.mark.parametrize("dt_value", nystrom_dt_values)
@pytest.mark.parametrize("method,stepper_type", nystrom_schemes)
def test_bilinear_jacobian_uses_stage_state_nystrom(method, stepper_type, dt_value):
    F, Jhand, t, dt, u, ut, x = wave_reaction_problem(dt_value)

    auto = stepper_type(F, method, t, dt, u, ut)
    hand = stepper_type(F, method, t, dt, u, ut, J=Jhand)

    moved, diff = compare(auto.problem.J, hand.problem.J,
                          nystrom_stage_unknown(auto),
                          nystrom_stage_unknown(hand), x)

    assert moved > 1e-3
    assert diff < 1e-12


def test_bilinear_preconditioner_uses_stage_state():
    """Jp is substituted the same way J is."""
    F, Jhand, t, dt, u, x = heat_reaction_problem(1.0)

    auto = TimeStepper(F, RadauIIA(2), t, dt, u, stage_type="deriv")
    hand = TimeStepper(F, RadauIIA(2), t, dt, u, Jp=Jhand, stage_type="deriv")

    moved, diff = compare(auto.problem.J, hand.problem.Jp,
                          auto.stages, hand.stages, x)

    assert moved > 1e-3
    assert diff < 1e-12


def test_bilinear_preconditioner_through_aux_pc():
    """A bilinear Jp reaches the auxiliary operator PC as a bilinear form."""
    F, Jhand, t, dt, u, x = heat_reaction_problem(0.5)
    V = u.function_space()
    u_init = Function(V).assign(u)

    aux_parameters = {
        "snes_rtol": 1e-12,
        "snes_atol": 1e-14,
        "mat_type": "matfree",
        "ksp_type": "gmres",
        "ksp_rtol": 1e-12,
        "ksp_atol": 1e-14,
        "pc_type": "python",
        "pc_python_type": "irksome.IRKAuxiliaryOperatorPC",
        "aux_pc_type": "lu",
    }
    stepper = TimeStepper(F, RadauIIA(2), t, dt, u, stage_type="deriv",
                          Jp=Jhand, solver_parameters=aux_parameters)
    stepper.advance()
    u_aux = Function(V).assign(u)

    direct_parameters = {
        "snes_rtol": 1e-12,
        "snes_atol": 1e-14,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    u.assign(u_init)
    reference = TimeStepper(F, RadauIIA(2), t, dt, u, stage_type="deriv",
                            solver_parameters=direct_parameters)
    reference.advance()

    assert errornorm(u, u_aux) < 1e-10


@pytest.mark.parametrize("stage_type", ("deriv", "value"))
def test_bilinear_residual_reads_u0_as_the_stage_state(stage_type):
    """Bare u0 is the stage state inside a bilinear F; lag holds it at t_n."""
    msh = UnitIntervalMesh(4)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    t = MC.Constant(0.0)
    dt = MC.Constant(1.0)
    (x,) = SpatialCoordinate(msh)
    v = TestFunction(V)
    du = TrialFunction(V)
    dxq = dx(degree=6)

    def rhs_movement(source):
        u = Function(V).interpolate(1.0 + 0.5 * cos(pi * x))
        # the source reads the unknown, so it lands in the right-hand side
        F = (inner(Dt(du), v) * dxq
             + inner(grad(du), grad(v)) * dxq
             - inner(source(u) ** 2, v) * dxq)
        stepper = TimeStepper(F, RadauIIA(2), t, dt, u, stage_type=stage_type)

        Fbig, _ = stepper.get_form_and_bcs(stepper.stages)
        _, L = ufl.system(Fbig)

        def rhs():
            b = assemble(L)
            return np.concatenate([np.asarray(bi.dat.data_ro).ravel()
                                   for bi in b.subfunctions])

        b_zero = rhs()
        perturb(stepper.stages, x)
        return np.linalg.norm(rhs() - b_zero) / np.linalg.norm(b_zero)

    bare = rhs_movement(lambda w: w)
    lagged = rhs_movement(lag)
    print(f"{stage_type}: rhs moved bare {bare:.3e}, lagged {lagged:.3e}")

    assert bare > 1e-3
    assert lagged < 1e-12


def test_lag_holds_a_coefficient_at_the_old_step():
    """A coefficient wrapped in lag stays at t_n, as a Picard operator needs."""
    F, _, t, dt, u, x = heat_reaction_problem(1.0)
    v = TestFunction(u.function_space())
    du = TrialFunction(u.function_space())
    dxq = dx(degree=6)
    Jlag = (inner(Dt(du), v) * dxq
            + inner(grad(du), grad(v)) * dxq
            + inner(3 * lag(u) ** 2 * du, v) * dxq)

    stepper = TimeStepper(F, RadauIIA(2), t, dt, u, J=Jlag, stage_type="deriv")

    A_zero = petsc_mat(stepper.problem.J)
    perturb(stepper.stages, x)
    frozen = reldiff(petsc_mat(stepper.problem.J), A_zero)
    print(f"lagged Jacobian moved {frozen:.3e}")

    assert frozen < 1e-12


def test_bilinear_jacobian_nonlinear_time_derivative():
    """Dt(u0) survives when the form is nonlinear in the time derivative.

    Differentiating inner(Dt(u)*u, v) gives inner(Dt(du)*u + Dt(u)*du, v), so
    the Jacobian carries a time derivative on both symbols.  The stage value
    formulation rejects such a form outright, so this covers the stage
    derivative one.
    """
    msh = UnitIntervalMesh(4)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    t, dt = MC.Constant(0.0), MC.Constant(1.0)
    (x,) = SpatialCoordinate(msh)
    dxq = dx(degree=6)

    u = Function(V).interpolate(1.0 + 0.5 * cos(pi * x))
    v = TestFunction(V)
    du = TrialFunction(V)
    F = inner(Dt(u) * u, v) * dxq + inner(grad(u), grad(v)) * dxq
    Jhand = (inner(Dt(du) * u + Dt(u) * du, v) * dxq
             + inner(grad(du), grad(v)) * dxq)

    auto = TimeStepper(F, RadauIIA(2), t, dt, u, stage_type="deriv")
    hand = TimeStepper(F, RadauIIA(2), t, dt, u, stage_type="deriv", J=Jhand)

    moved, diff = compare(auto.problem.J, hand.problem.J,
                          auto.stages, hand.stages, x)

    assert moved > 1e-3
    assert diff < 1e-12
