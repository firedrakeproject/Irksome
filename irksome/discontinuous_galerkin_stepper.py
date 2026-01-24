from FIAT import (Bernstein,
                  DiscontinuousLagrange, Legendre,
                  GaussLobattoLegendre, GaussRadau)
from ufl import as_ufl, as_tensor
from ufl.algorithms.analysis import has_type
from .base_time_stepper import StageCoupledTimeStepper
from .bcs import stage2spaces4bc
from .deriv import TimeDerivative, expand_time_derivatives
from .estimate_degrees import TimeDegreeEstimator, get_degree_mapping
from .labeling import split_quadrature, as_form
from .manipulation import extract_terms, strip_dt_form
from .scheme import create_time_quadrature, ufc_line
from .tools import dot, reshape, replace, vecconst
import numpy as np
from firedrake import TestFunction


def getElement(basis_type, order):
    if basis_type is not None:
        basis_type = basis_type.lower()
    if basis_type == "lobatto":
        if order == 0:
            raise ValueError("Lobatto test element needs degree > 0")
        return GaussLobattoLegendre(ufc_line, order)
    elif basis_type == "radau":
        return GaussRadau(ufc_line, order)
    elif basis_type == "integral":
        return Legendre(ufc_line, order)
    elif basis_type == "bernstein":
        if order == 0:
            return DiscontinuousLagrange(ufc_line, order)
        else:
            return Bernstein(ufc_line, order)
    else:
        # Let recursivenodes handle the general case
        variant = None if basis_type == "lagrange" else basis_type
        return DiscontinuousLagrange(ufc_line, order, variant=variant)


def getTermDiscGalerkin(F, L, Q, t, dt, u0, stages, test):
    # preprocess time derivatives
    F = expand_time_derivatives(F, t=t, timedep_coeffs=(u0,))
    v, = F.arguments()
    V = v.function_space()
    assert V == u0.function_space()

    num_stages = L.space_dimension()
    qpts = Q.get_points()
    qwts = Q.get_weights()
    assert np.size(qpts) >= num_stages

    tabulate_basis = L.tabulate(1, qpts)
    basis_vals = tabulate_basis[(0,)]
    basis_dvals = tabulate_basis[(1,)]
    basis_vals_w = np.multiply(basis_vals, qwts)

    trial_vals = vecconst(basis_vals)
    trial_dvals = vecconst(basis_dvals)
    test_vals_w = vecconst(basis_vals_w)
    qpts = vecconst(qpts.reshape((-1,)))

    # set up the pieces we need to work with to do our substitutions
    v_np = reshape(test, (num_stages, *u0.ufl_shape))
    u_np = reshape(stages, (num_stages, *u0.ufl_shape))
    vsub = dot(test_vals_w.T, v_np)
    usub = dot(trial_vals.T, u_np)
    dtu0sub = dot(trial_dvals.T, u_np)

    if has_type(F, TimeDerivative):
        split_form = extract_terms(F)
        F_dtless = strip_dt_form(split_form.time)
        F_remainder = split_form.remainder

        # Jump terms
        L_at_0 = vecconst(L.tabulate(0, (0.0,))[(0,)])
        u_at_0 = L_at_0 @ u_np
        v_at_0 = L_at_0 @ v_np
        repl = {u0: u_at_0 - u0,
                v: v_at_0}
        Fnew = replace(F_dtless, repl)

        # Terms with time derivatives
        for q in range(len(qpts)):
            repl = {t: t + qpts[q] * dt,
                    v: vsub[q] * dt,
                    u0: dtu0sub[q] / dt}
            Fnew += replace(F_dtless, repl)
    else:
        Fnew = 0
        F_remainder = F

    # Handle the rest of the terms
    for q in range(len(qpts)):
        repl = {t: t + qpts[q] * dt,
                v: vsub[q] * dt,
                u0: usub[q]}
        Fnew += replace(F_remainder, repl)
    return Fnew


def getFormDiscGalerkin(F, L, Qdefault, t, dt, u0, stages, bcs=None):
    """Given a time-dependent variational form, trial and test spaces, and
    a quadrature rule, produce UFL for the Discontinuous Galerkin-in-Time method.

    :arg F: UFL form for the semidiscrete ODE/DAE
    :arg L: A :class:`FIAT.FiniteElement` for the test and trial functions in time
    :arg Qdefault: A :class:`FIAT.QuadratureRule` for the time integration
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg stages: a :class:`Function` representing the stages to be solved for.
    :arg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
         containing (possibly time-dependent) boundary conditions imposed
         on the system.

    On output, we return a tuple consisting of two parts:

       - Fnew, the :class:`Form` corresponding to the DG-in-Time discretized problem
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the Galerkin-in-time solution,
    """
    num_stages = L.space_dimension()

    V = u0.function_space()
    Vbig = stages.function_space()
    test = TestFunction(Vbig)

    degree_mapping = get_degree_mapping(as_form(F), L.degree(), L.degree(), t=t, timedep_coeffs=(u0,))
    degree_estimator = TimeDegreeEstimator(degree_mapping=degree_mapping)

    splitting = split_quadrature(F, degree_estimator=degree_estimator, Qdefault=Qdefault)
    Fnew = sum(getTermDiscGalerkin(Fcur, L, Q, t, dt, u0, stages, test)
               for Q, Fcur in splitting.items())

    # Oh, honey, is it the boundary conditions?
    bcsnew = []
    if bcs:
        if Qdefault is None or isinstance(Qdefault, str):
            # create a quadrature for the boundary conditions
            bc_degree = max(max(degree_estimator(as_ufl(bc._original_arg)) for bc in bcs), L.degree())
            Qdefault = create_time_quadrature(bc_degree + L.degree(), scheme=Qdefault)

        qpts = Qdefault.get_points()
        basis_vals = L.tabulate(0, qpts)[(0,)]
        basis_vals_w = np.multiply(basis_vals, Qdefault.get_weights())
        # mass matrix for BC, based on default quadrature rule
        mmat = basis_vals_w @ basis_vals.T
        # L2 projector
        proj = vecconst(np.linalg.solve(mmat, basis_vals_w))
        qpts = vecconst(qpts.reshape((-1,)))

        for bc in bcs:
            g0 = as_ufl(bc._original_arg)
            gq = np.array([replace(g0, {t: t + c*dt}) for c in qpts])
            g_np = proj @ gq
            for i in range(num_stages):
                Vbigi = stage2spaces4bc(bc, V, Vbig, i)
                bcsnew.append(bc.reconstruct(V=Vbigi, g=as_tensor(g_np[i])))
    return Fnew, bcsnew


class DiscontinuousGalerkinTimeStepper(StageCoupledTimeStepper):
    """Front-end class for advancing a time-dependent PDE via a Discontinuous Galerkin
    in time method

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg scheme: a :class:`DiscontinuousGalerkinScheme` instance describing the order,
         basis type, and default quadrature scheme.
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing
            the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the method.
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    :arg appctx: An optional :class:`dict` containing application context.
            This gets included with particular things that Irksome will
            pass into the nonlinear solver so that, say, user-defined preconditioners
            have access to it.
    :arg nullspace: A list of tuples of the form (index, VSB) where
            index is an index into the function space associated with
            `u` and VSB is a :class: `firedrake.VectorSpaceBasis`
            instance to be passed to a
            `firedrake.MixedVectorSpaceBasis` over the larger space
            associated with the Runge-Kutta method
    """
    def __init__(self, F, scheme, t, dt, u0, bcs=None, basis_type=None,
                 quadrature=None, **kwargs):
        order = self.order = scheme.order
        assert order >= 0, "DG must be order >= 0"

        self.basis_type = basis_type = scheme.basis_type

        V = u0.function_space()
        self.num_fields = len(V)

        self.el = getElement(basis_type, order)

        quad_degree = scheme.quadrature_degree
        if quad_degree is None:
            quad_degree = 2*self.el.degree()
        quad_scheme = scheme.quadrature_scheme
        if quad_scheme is None and isinstance(self.el, GaussRadau):
            quad_scheme = "radau"

        if quad_degree == "auto":
            quadrature = quad_scheme
        else:
            quadrature = create_time_quadrature(quad_degree, scheme=quad_scheme)

        self.quadrature = quadrature

        num_stages = order+1

        self.update_b = vecconst(self.el.tabulate(0, (1.0,))[(0,)])

        super().__init__(F, t, dt, u0, num_stages, bcs=bcs, **kwargs)

    def get_form_and_bcs(self, stages, F=None, bcs=None, basis_type=None, order=None, quadrature=None):
        if bcs is None:
            bcs = self.orig_bcs
        if basis_type is None:
            basis_type = self.basis_type
        if order is None:
            order = self.order
        if basis_type == self.basis_type and order == self.order:
            el = self.el
        else:
            el = getElement(basis_type, order)
        return getFormDiscGalerkin(F or self.F,
                                   el,
                                   quadrature or self.quadrature,
                                   self.t, self.dt, self.u0, stages, bcs)

    def _update(self):
        stages_np = np.array(self.stages.subfunctions, dtype=object)
        for i, u0bit in enumerate(self.u0.subfunctions):
            u0bit.assign(stages_np[i::self.num_fields] @ self.update_b)
