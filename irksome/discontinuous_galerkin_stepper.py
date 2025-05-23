from FIAT import (Bernstein, DiscontinuousElement,
                  DiscontinuousLagrange,
                  Legendre,
                  make_quadrature, ufc_simplex)
from ufl.constantvalue import as_ufl
from .base_time_stepper import StageCoupledTimeStepper
from .bcs import stage2spaces4bc
from .deriv import expand_time_derivatives
from .manipulation import extract_terms, strip_dt_form
from .tools import replace, vecconst
import numpy as np
from firedrake import TestFunction


ufc_line = ufc_simplex(1)


def getElement(basis_type, order):
    if order == 0:
        el = DiscontinuousLagrange(ufc_line, order)
    elif basis_type == "Bernstein":
        el = DiscontinuousElement(Bernstein(ufc_line, order))
    elif basis_type == "integral":
        el = Legendre(ufc_line, order)
    else:
        # Let recursivenodes handle the general case
        variant = None if basis_type == "Lagrange" else basis_type
        el = DiscontinuousLagrange(ufc_line, order, variant=variant)
    return el


def getFormDiscGalerkin(F, L, Q, t, dt, u0, stages, bcs=None):

    """Given a time-dependent variational form, trial and test spaces, and
    a quadrature rule, produce UFL for the Discontinuous Galerkin-in-Time method.

    :arg F: UFL form for the semidiscrete ODE/DAE
    :arg L: A :class:`FIAT.FiniteElement` for the test and trial functions in time
    :arg Q: A :class:`FIAT.QuadratureRule` for the time integration
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
    :arg nullspace: A list of tuples of the form (index, VSB) where
         index is an index into the function space associated with `u`
         and VSB is a :class: `firedrake.VectorSpaceBasis` instance to
         be passed to a `firedrake.MixedVectorSpaceBasis` over the
         larger space associated with the Runge-Kutta method

    On output, we return a tuple consisting of four parts:

       - Fnew, the :class:`Form` corresponding to the DG-in-Time discretized problem
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the Galerkin-in-time solution,
    """
    assert Q.ref_el.get_spatial_dimension() == 1
    assert L.get_reference_element() == Q.ref_el

    # preprocess time derivatives
    F = expand_time_derivatives(F, t=t, timedep_coeffs=(u0,))
    v, = F.arguments()
    V = v.function_space()
    assert V == u0.function_space()

    num_stages = L.space_dimension()
    Vbig = stages.function_space()
    test = TestFunction(Vbig)
    qpts = Q.get_points()
    qwts = Q.get_weights()

    tabulate_basis = L.tabulate(1, qpts)
    basis_vals = tabulate_basis[(0,)]
    basis_dvals = tabulate_basis[(1,)]
    basis_vals_w = np.multiply(basis_vals, qwts)

    # mass matrix later for BC
    mmat = basis_vals_w @ basis_vals.T
    # L2 projector
    proj = vecconst(np.linalg.solve(mmat, basis_vals_w))

    trial_vals = vecconst(basis_vals)
    trial_dvals = vecconst(basis_dvals)
    test_vals_w = vecconst(basis_vals_w)
    qpts = vecconst(qpts.reshape((-1,)))

    split_form = extract_terms(F)
    F_dtless = strip_dt_form(split_form.time)
    F_remainder = split_form.remainder

    # set up the pieces we need to work with to do our substitutions
    v_np = np.reshape(test, (num_stages, *u0.ufl_shape))
    u_np = np.reshape(stages, (num_stages, *u0.ufl_shape))
    vsub = test_vals_w.T @ v_np
    usub = trial_vals.T @ u_np
    dtu0sub = trial_dvals.T @ u_np

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

    # Handle the rest of the terms
    for q in range(len(qpts)):
        repl = {t: t + qpts[q] * dt,
                v: vsub[q] * dt,
                u0: usub[q]}
        Fnew += replace(F_remainder, repl)

    # Oh, honey, is it the boundary conditions?
    if bcs is None:
        bcs = []
    bcsnew = []
    for bc in bcs:
        g0 = as_ufl(bc._original_arg)
        Vg_np = np.array([replace(g0, {t: t + c*dt}) for c in qpts])
        g_np = proj @ Vg_np
        for i in range(num_stages):
            Vbigi = stage2spaces4bc(bc, V, Vbig, i)
            bcsnew.append(bc.reconstruct(V=Vbigi, g=g_np[i]))

    return Fnew, bcsnew


class DiscontinuousGalerkinTimeStepper(StageCoupledTimeStepper):
    """Front-end class for advancing a time-dependent PDE via a Discontinuous Galerkin
    in time method

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg order: an integer indicating the order of the DG space to use
         (with order == 0 corresponding to DG(0)-in-time)
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
    :arg basis_type: A string indicating the finite element family (either
            `'Lagrange'` or `'Bernstein'`) or the Lagrange variant for the
            test/trial spaces. Defaults to equispaced Lagrange elements.
    :arg quadrature: A :class:`FIAT.QuadratureRule` indicating the quadrature
            to be used in time, defaulting to GL with order+1 points
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
    def __init__(self, F, order, t, dt, u0, bcs=None, basis_type=None,
                 quadrature=None, **kwargs):
        assert order >= 0
        self.order = order
        self.basis_type = basis_type

        V = u0.function_space()
        self.num_fields = len(V)

        self.el = getElement(basis_type, order)

        if quadrature is None:
            quadrature = make_quadrature(ufc_line, order+1)
        self.quadrature = quadrature
        assert np.size(quadrature.get_points()) >= order+1

        num_stages = order+1

        self.update_b = vecconst(self.el.tabulate(0, (1.0,))[(0,)])

        super().__init__(F, t, dt, u0, num_stages, bcs=bcs, **kwargs)

    def get_form_and_bcs(self, stages, basis_type=None, order=None, quadrature=None, F=None):
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
                                   self.t, self.dt, self.u0, stages, self.orig_bcs)

    def _update(self):
        stages_np = np.array(self.stages.subfunctions, dtype=object)
        for i, u0bit in enumerate(self.u0.subfunctions):
            u0bit.assign(stages_np[i::self.num_fields] @ self.update_b)
