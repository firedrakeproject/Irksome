"""A bilinear J must be evaluated at the stage state.

At the initial guess the stage state equals u0, so the stages are perturbed
before assembling.
"""
import numpy as np
import pytest
import ufl
from firedrake import (Constant, Function, FunctionSpace, SpatialCoordinate, TestFunction,
                       TrialFunction, UnitIntervalMesh, assemble, cos, derivative, dx,
                       errornorm, grad, inner, pi)
from irksome import (Alexander, BDF, ContinuousPetrovGalerkinScheme,
                     DIRKNystromTimeStepper, DiscontinuousGalerkinScheme, Dt,
                     GalerkinCollocationScheme, GaussLegendre,
                     QinZhang, RadauIIA, StageDerivativeNystromTimeStepper,
                     TimeStepper, lag)
from irksome.dirk_stepper import DIRKTimeStepper
from irksome.multistep import MultistepTimeStepper


# quadrature_degree is pinned so that F and J use the same rule
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

# A second dt separates a missing or spurious factor of dt.  The Nystrom
# state depends on the stages through dt**2, so its second dt is above one.
dt_values = [1.0, 0.37]
nystrom_dt_values = [1.0, 2.5]

# how the J under test is given: built by Irksome, or derivative(F, u)
jacobians = ["default", "derivative"]

dxq = dx(degree=6)


def setup(dt_value):
    msh = UnitIntervalMesh(4)
    V = FunctionSpace(msh, "CG", 1)
    (x,) = SpatialCoordinate(msh)
    u = Function(V).interpolate(1.0 + 0.5 * cos(pi * x))
    return Constant(0.0), Constant(dt_value), u, TestFunction(V), TrialFunction(V), x


def heat_reaction_problem(dt_value):
    t, dt, u, v, du, x = setup(dt_value)
    F = (inner(Dt(u), v) * dxq
         + inner(grad(u), grad(v)) * dxq
         + inner(u ** 3, v) * dxq)
    Jhand = (inner(Dt(du), v) * dxq
             + inner(grad(du), grad(v)) * dxq
             + inner(3 * u ** 2 * du, v) * dxq)
    return F, Jhand, t, dt, u, x


def wave_reaction_problem(dt_value):
    # a nonzero ut also moves the Nystrom state away from u0
    t, dt, u, v, du, x = setup(dt_value)
    ut = Function(u.function_space()).interpolate(0.25 * cos(2 * pi * x))
    F = (inner(Dt(u, 2), v) * dxq
         + inner(grad(u), grad(v)) * dxq
         + inner(u ** 3, v) * dxq)
    Jhand = (inner(Dt(du, 2), v) * dxq
             + inner(grad(du), grad(v)) * dxq
             + inner(3 * u ** 2 * du, v) * dxq)
    return F, Jhand, t, dt, u, ut, x


def stage_unknown(stepper):
    """The Function the stage problem solves for, set up as for the first stage."""
    if isinstance(stepper, DIRKTimeStepper):
        k, g, a, c = stepper.kgac
        g.assign(stepper.u0)
        a.assign(float(stepper.butcher_tableau.A[0, 0]))
        c.assign(float(stepper.butcher_tableau.c[0]))
        return k
    if isinstance(stepper, DIRKNystromTimeStepper):
        k, g1, g2, a, abar, c = stepper.kgac
        g1.assign(stepper.u0 + stepper.ut0 * (stepper.CC[0] * stepper.dt))
        g2.assign(stepper.ut0)
        a.assign(float(stepper.AA[0, 0]))
        abar.assign(float(stepper.AAbar[0, 0]))
        c.assign(float(stepper.CC[0]))
        return k
    if isinstance(stepper, MultistepTimeStepper):
        return stepper.us[-1]
    return stepper.stages


def perturb(w, x):
    for i, wi in enumerate(w.subfunctions):
        wi.interpolate(0.6 * cos((i + 1) * pi * x) - 0.2)


def petsc_mat(form):
    return assemble(form, mat_type="aij").petscmat


def reldiff(A, B):
    D = A.copy()
    D.axpy(-1.0, B)
    return D.norm() / A.norm()


def compare(J_auto, J_hand, w_auto, w_hand, x):
    """Perturb the stages; return how far J_auto moved and how far it is from J_hand."""
    A_zero = petsc_mat(J_auto)
    perturb(w_auto, x)
    perturb(w_hand, x)
    A_auto = petsc_mat(J_auto)
    A_hand = petsc_mat(J_hand)
    return reldiff(A_auto, A_zero), reldiff(A_auto, A_hand)


@pytest.mark.parametrize("jacobian", jacobians)
@pytest.mark.parametrize("dt_value", dt_values)
@pytest.mark.parametrize("method,kwargs", schemes)
def test_jacobian(method, kwargs, dt_value, jacobian):
    F, Jhand, t, dt, u, x = heat_reaction_problem(dt_value)
    J = None
    if jacobian == "derivative":
        J = derivative(F, u)

    auto = TimeStepper(F, method, t, dt, u, J=J, **kwargs)
    hand = TimeStepper(F, method, t, dt, u, J=Jhand, **kwargs)

    moved, diff = compare(auto.problem.J, hand.problem.J,
                          stage_unknown(auto), stage_unknown(hand), x)
    # without movement the check below would also pass on a frozen J
    assert moved > 1e-3
    assert diff < 1e-12


@pytest.mark.parametrize("jacobian", jacobians)
@pytest.mark.parametrize("dt_value", nystrom_dt_values)
@pytest.mark.parametrize("method,stepper_type", nystrom_schemes)
def test_jacobian_nystrom(method, stepper_type, dt_value, jacobian):
    F, Jhand, t, dt, u, ut, x = wave_reaction_problem(dt_value)
    J = None
    if jacobian == "derivative":
        J = derivative(F, u)

    auto = stepper_type(F, method, t, dt, u, ut, J=J)
    hand = stepper_type(F, method, t, dt, u, ut, J=Jhand)

    moved, diff = compare(auto.problem.J, hand.problem.J,
                          stage_unknown(auto), stage_unknown(hand), x)
    assert moved > 1e-3
    assert diff < 1e-12


def test_jp():
    F, Jhand, t, dt, u, x = heat_reaction_problem(1.0)

    auto = TimeStepper(F, RadauIIA(2), t, dt, u, stage_type="deriv")
    hand = TimeStepper(F, RadauIIA(2), t, dt, u, Jp=Jhand, stage_type="deriv")

    moved, diff = compare(auto.problem.J, hand.problem.Jp,
                          auto.stages, hand.stages, x)
    assert moved > 1e-3
    assert diff < 1e-12


def test_jp_aux_pc():
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
def test_bilinear_rhs(stage_type):
    """Bare u0 is the stage state inside a bilinear F; lag holds it at t_n."""

    def rhs_movement(source):
        t, dt, u, v, du, x = setup(1.0)
        F = (inner(Dt(du), v) * dxq
             + inner(grad(du), grad(v)) * dxq
             - inner(source(u) ** 2, v) * dxq)
        stepper = TimeStepper(F, RadauIIA(2), t, dt, u, stage_type=stage_type)

        Fbig, _ = stepper.get_form_and_bcs(stepper.stages)
        _, L = ufl.system(Fbig)

        def rhs():
            b = assemble(L)
            return np.concatenate([bi.dat.data_ro.ravel() for bi in b.subfunctions])

        b_zero = rhs()
        perturb(stepper.stages, x)
        return np.linalg.norm(rhs() - b_zero) / np.linalg.norm(b_zero)

    assert rhs_movement(lambda w: w) > 1e-3
    assert rhs_movement(lag) < 1e-12


def test_lag():
    F, _, t, dt, u, x = heat_reaction_problem(1.0)
    v = TestFunction(u.function_space())
    du = TrialFunction(u.function_space())
    Jlag = (inner(Dt(du), v) * dxq
            + inner(grad(du), grad(v)) * dxq
            + inner(3 * lag(u) ** 2 * du, v) * dxq)

    stepper = TimeStepper(F, RadauIIA(2), t, dt, u, J=Jlag, stage_type="deriv")

    A_zero = petsc_mat(stepper.problem.J)
    perturb(stepper.stages, x)
    assert reldiff(petsc_mat(stepper.problem.J), A_zero) < 1e-12


def test_jacobian_nonlinear_dt():
    # stage value rejects a form nonlinear in Dt, so only stage derivative is covered
    t, dt, u, v, du, x = setup(1.0)
    F = inner(Dt(u) * u, v) * dxq + inner(grad(u), grad(v)) * dxq
    Jhand = (inner(Dt(du) * u + Dt(u) * du, v) * dxq
             + inner(grad(du), grad(v)) * dxq)

    auto = TimeStepper(F, RadauIIA(2), t, dt, u, stage_type="deriv")
    hand = TimeStepper(F, RadauIIA(2), t, dt, u, stage_type="deriv", J=Jhand)

    moved, diff = compare(auto.problem.J, hand.problem.J,
                          auto.stages, hand.stages, x)
    assert moved > 1e-3
    assert diff < 1e-12
