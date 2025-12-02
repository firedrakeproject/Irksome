from FIAT import (Bernstein, DiscontinuousLagrange,
                  GaussRadau, IntegratedLegendre, Lagrange,
                  NodalEnrichedElement, RestrictedElement)
from ufl.classes import Zero
from ufl import as_ufl, as_tensor

from .base_time_stepper import StageCoupledTimeStepper
from .bcs import bc2space, extract_bcs, stage2spaces4bc
from .deriv import TimeDerivative, expand_time_derivatives
from .labeling import split_quadrature
from .scheme import GalerkinCollocationScheme, create_time_quadrature, ufc_line
from .tools import AI, IA, dot, fields_to_components, reshape, replace, vecconst
from .discontinuous_galerkin_stepper import getElement as getTestElement
from .integrated_lagrange import IntegratedLagrange

from .ButcherTableaux import CollocationButcherTableau
from .stage_derivative import getForm
from .stage_value import getFormStage

import numpy as np
from firedrake import TestFunction, Constant
from firedrake.bcs import EquationBCSplit


def getTrialElement(basis_type, order):
    if basis_type is not None:
        basis_type = basis_type.lower()
    if basis_type == "bernstein":
        return Bernstein(ufc_line, order)
    elif basis_type == "integral":
        return IntegratedLegendre(ufc_line, order)
    else:
        # Let recursivenodes handle the general case
        variant = None if basis_type == "lagrange" else basis_type
        return Lagrange(ufc_line, order, variant=variant)


def getElements(basis_type, order):
    if isinstance(basis_type, (tuple, list)):
        trial_type, test_type = basis_type
    else:
        trial_type = basis_type
        test_type = basis_type

    L_test = getTestElement(test_type, order-1)
    # Deal with GalerkinCollocationScheme
    if trial_type == "deriv":
        L_trial = IntegratedLagrange(L_test)
    elif trial_type == "value":
        if len(L_test.entity_dofs()[0][0]) != 0:
            # Confluent case
            H = IntegratedLagrange(L_test)
            indices = [k for k, node in enumerate(H.dual)
                       if (0.0,) in node.deriv_dict]
            R = RestrictedElement(H, indices=indices)
        else:
            CG = getTrialElement("spectral", order)
            R = RestrictedElement(CG, indices=CG.entity_dofs()[0][0])

        L_trial = NodalEnrichedElement(R, L_test)
    else:
        L_trial = getTrialElement(trial_type, order)
    return L_trial, L_test


def getTermGalerkin(F, L_trial, L_test, Q, t, dt, u0, stages, test, aux_indices):
    # preprocess time derivatives
    F = expand_time_derivatives(F, t=t, timedep_coeffs=(u0,))
    v, = F.arguments()
    V = v.function_space()
    assert V == u0.function_space()
    i0, = L_trial.entity_dofs()[0][0]

    qpts = Q.get_points()
    qwts = Q.get_weights()
    tabulate_trials = L_trial.tabulate(1, qpts)
    trial_vals = tabulate_trials[(0,)]
    trial_dvals = tabulate_trials[(1,)]
    test_vals = L_test.tabulate(0, qpts)[(0,)]
    test_vals_w = np.multiply(test_vals, qwts)

    trial_vals = vecconst(trial_vals)
    trial_dvals = vecconst(trial_dvals)
    test_vals = vecconst(test_vals)
    test_vals_w = vecconst(test_vals_w)
    qpts = vecconst(np.reshape(qpts, (-1,)))

    # set up the pieces we need to work with to do our substitutions
    v_np = reshape(test, (-1, *v.ufl_shape))
    w_np = reshape(stages, (-1, *u0.ufl_shape))
    u_np = np.insert(w_np, i0, reshape(u0, (1, *u0.ufl_shape)), axis=0)

    vsub = dot(test_vals_w.T, v_np)
    usub = dot(trial_vals.T, u_np)
    dtu0sub = dot(trial_dvals.T, u_np)
    dtu0 = TimeDerivative(u0)

    # discretize the auxiliary fields in the DG test space
    if aux_indices is not None:
        aux_components = fields_to_components(V, aux_indices)
        usub[:, aux_components] = dot(test_vals.T, w_np[:, aux_components])

    # now loop over quadrature points
    repl = {}
    for q in range(len(qpts)):
        repl[q] = {t: t + qpts[q] * dt,
                   v: vsub[q] * dt,
                   u0: usub[q],
                   dtu0: dtu0sub[q] / dt}
    Fnew = sum(replace(F, repl[q]) for q in repl)
    return Fnew


def getFormGalerkin(F, L_trial, L_test, Qdefault, t, dt, u0, stages, bcs=None, bc_type=None, aux_indices=None):
    """Given a time-dependent variational form, trial and test spaces, and
    a quadrature rule, produce UFL for the Galerkin-in-Time method.

    :arg F: UFL form for the semidiscrete ODE/DAE
    :arg L_trial: A :class:`FIAT.FiniteElement` for the trial functions in time
    :arg L_test: A :class:`FIAT.FinteElement` for the test functions in time
    :arg Qdefault: A :class:`FIAT.QuadratureRule` for the time integration.
         This rule will be used for all terms in the semidiscrete
         variational form that aren't specifically tagged with another
         quadrature rule.
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg stages: a :class:`Function` representing the stages to be solved for.
    :kwarg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
         containing (possibly time-dependent) boundary conditions imposed
         on the system.
    :kwarg bc_type: How to manipulate the strongly-enforced boundary
         conditions to derive the stage boundary conditions.  Should
         be a string, either "DAE", which implements BCs as
         constraints in the style of a differential-algebraic
         equation, or "ODE", which evaluates the temporal degrees of
         freedom of the boundary data.
         Currently there is no support for `firedrake.EquationBC`.
    :kwarg aux_indices: a list of field indices to be discretized in the test space
         rather than trial space.

    :returns: a 2-tuple of
       - `Fnew`, the :class:`Form` corresponding to the Galerkin-in-Time discretized problem
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the Galerkin-in-time solution
    """
    if bc_type is None:
        bc_type = "DAE"

    assert L_test.get_reference_element() == Qdefault.ref_el
    assert L_trial.get_reference_element() == Qdefault.ref_el
    assert Qdefault.ref_el.get_spatial_dimension() == 1
    assert L_trial.space_dimension() == L_test.space_dimension() + 1

    Vbig = stages.function_space()
    test = TestFunction(Vbig)

    splitting = split_quadrature(F, Qdefault=Qdefault)
    Fnew = sum(getTermGalerkin(Fcur, L_trial, L_test, Q, t, dt, u0, stages, test, aux_indices)
               for Q, Fcur in splitting.items())

    # Oh, honey, is it the boundary conditions?
    V = u0.function_space()
    if bcs is None:
        bcs = []
    bcsnew = []

    # list of dictionaries mapping time coordinates to weights to evaluate DOFs
    test_dicts = [{Constant(c): Constant(sum(w for (w, *_) in wts))
                  for (c,), wts in node.pt_dict.items()} for node in L_test.dual_basis()]

    def evaluate_dof(dof, g):
        if isinstance(g, Zero):
            return g
        return sum(replace(g, {t: t + c * dt}) * w for c, w in dof.items())

    aux_bc_data = lambda bc: [evaluate_dof(dof, as_ufl(bc._original_arg)) for dof in test_dicts]

    i0, = L_trial.entity_dofs()[0][0]
    if bc_type == "ODE":
        nodes = list(L_trial.dual_basis())
        del nodes[i0]
        trial_dicts = [{Constant(c): Constant(sum(w for (w, *_) in wts))
                       for (c,), wts in node.pt_dict.items()} for node in nodes]
        dtrial_dicts = [{Constant(c): Constant(sum(w for (w, *_) in wts))
                        for (c,), wts in node.deriv_dict.items()} for node in nodes]

        def evaluate_trial_dof(bc, value_dict, deriv_dict):
            g = as_ufl(bc._original_arg)
            gi = evaluate_dof(value_dict, g)
            if deriv_dict:
                gt = expand_time_derivatives(TimeDerivative(g), t=t, timedep_coeffs=(u0,))
                gi += dt * evaluate_dof(deriv_dict, gt)
            return gi

        bc_data = lambda bc: [evaluate_trial_dof(bc, pt, dpt) for pt, dpt in zip(trial_dicts, dtrial_dicts)]
    elif bc_type == "DAE":
        # mass-ish matrix for BC, based on default quadrature rule
        qpts = Qdefault.get_points()
        qwts = Qdefault.get_weights()
        trial_vals = L_trial.tabulate(0, qpts)[(0,)]
        test_vals = L_test.tabulate(0, qpts)[(0,)]
        test_vals_w = np.multiply(test_vals, qwts)

        mmat = test_vals_w @ np.delete(trial_vals, i0, axis=0).T
        try:
            trial_proj = vecconst(np.linalg.solve(mmat, test_vals_w))
        except np.linalg.LinAlgError:
            raise NotImplementedError("Cannot have DAE BCs for this basis type")
        trial_vals0 = vecconst(trial_vals[i0])
        qpts = vecconst(qpts.reshape((-1,)))

        def bc_data(bc):
            g = as_ufl(bc._original_arg)
            if isinstance(g, Zero):
                return [g]*len(test_dicts)
            ucur = bc2space(bc, u0)
            gq = np.array([replace(g, {t: t + q * dt}) - phi0 * ucur
                           for phi0, q in zip(trial_vals0, qpts)])
            return dot(trial_proj, gq)
    else:
        raise ValueError(f"Unrecognised bc_type: {bc_type}")

    num_stages = L_test.space_dimension()
    bcs = extract_bcs(bcs)
    for bc in bcs:
        if isinstance(bc, EquationBCSplit):
            raise NotImplementedError("EquationBC not implemented for Galerkin-in-Time")
        if aux_indices and bc.function_space().index in aux_indices:
            g_np = aux_bc_data(bc)
        else:
            g_np = bc_data(bc)
        for i in range(num_stages):
            Vbigi = stage2spaces4bc(bc, V, Vbig, i)
            bcsnew.append(bc.reconstruct(V=Vbigi, g=as_tensor(g_np[i])))
    return Fnew, bcsnew


class ContinuousPetrovGalerkinTimeStepper(StageCoupledTimeStepper):
    """Front-end class for advancing a time-dependent PDE via a Galerkin
    in time method

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg scheme: :class:`ContinuousPetrovGalerkinScheme` encoding the order,
         basis type, and default quadrature rule of the method.
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :kwarg bcs: An iterable of :class:`firedrake.DirichletBC` containing
            the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the method.
    :kwarg bc_type: How to manipulate the strongly-enforced boundary
         conditions to derive the stage boundary conditions.  Should
         be a string, either "DAE", which implements BCs as
         constraints in the style of a differential-algebraic
         equation, or "ODE", which evaluates the temporal degrees of
         freedom of the boundary data.
         Currently there is no support for `firedrake.EquationBC`.
    :kwarg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    :kwarg appctx: An optional :class:`dict` containing application context.
            This gets included with particular things that Irksome will
            pass into the nonlinear solver so that, say, user-defined preconditioners
            have access to it.
    :kwarg nullspace: A list of tuples of the form (index, VSB) where
            index is an index into the function space associated with
            `u` and VSB is a :class: `firedrake.VectorSpaceBasis`
            instance to be passed to a
            `firedrake.MixedVectorSpaceBasis` over the larger space
            associated with the Runge-Kutta method
    :kwarg aux_indices: a list of field indices to be discretized in the test space
            rather than trial space.
    """
    def __init__(self, F, scheme, t, dt, u0, bcs=None,
                 bc_type=None, aux_indices=None, **kwargs):
        self.order = scheme.order
        self.basis_type = scheme.basis_type

        V = u0.function_space()
        self.num_fields = len(V)

        self.trial_el, self.test_el = getElements(self.basis_type, self.order)
        num_stages = self.test_el.space_dimension()

        quad_degree = scheme.quadrature_degree
        if quad_degree is None:
            quad_degree = self.trial_el.degree() + self.test_el.degree()
        quad_scheme = scheme.quadrature_scheme
        if quad_scheme is None and isinstance(self.test_el, GaussRadau):
            quad_scheme = "radau"
        quadrature = create_time_quadrature(quad_degree, scheme=quad_scheme)
        assert np.size(quadrature.get_points()) >= num_stages

        self.quadrature = quadrature
        self.aux_indices = aux_indices
        if isinstance(scheme, GalerkinCollocationScheme):
            self.butcher_tableau = CollocationButcherTableau(self.test_el, None)
        else:
            self.butcher_tableau = None

        super().__init__(F, t, dt, u0, num_stages, bcs=bcs, bc_type=bc_type, **kwargs)
        self.set_initial_guess()
        self.set_update_expressions()

    def get_form_and_bcs(self, stages, F=None, u0=None, bcs=None,
                         tableau=None, basis_type=None, order=None,
                         quadrature=None, aux_indices=None):
        F = F or self.F
        u0 = u0 or self.u0
        if bcs is None:
            bcs = self.orig_bcs
        if basis_type is None:
            basis_type = self.basis_type
        aux_indices = aux_indices or self.aux_indices

        if tableau is not None:
            # Galerkin collocation is equivalent to an IRK up to row scaling
            row_scale = tableau.b

            def scaledIA(A):
                # For stage-value the splitting exposes row scaling
                A1, A2 = IA(A)
                np.multiply(A1, row_scale, out=A1)
                np.multiply(1/row_scale, A2, out=A2)
                return A1, A2

            # Construct the equivalent IRK stage residual
            trial_type, test_type = basis_type
            if trial_type == "value":
                splitting = scaledIA
                get_irk_form = getFormStage
            elif trial_type == "deriv":
                splitting = AI
                get_irk_form = getForm
            else:
                raise ValueError("Expecting a GalerkinCollocationScheme")

            Fnew, bcnew = get_irk_form(F, tableau, self.t, self.dt, u0, stages,
                                       bcs=bcs, splitting=splitting, aux_indices=aux_indices)

            if splitting != scaledIA:
                v0, = F.arguments()
                test, = Fnew.arguments()
                test_np = reshape(test, (-1, *v0.ufl_shape))
                test_np = np.multiply(vecconst(row_scale).reshape(-1, *(1,)*len(v0.ufl_shape)), test_np)
                test_np = test_np.reshape(test.ufl_shape)
                Fnew = replace(Fnew, {test: test_np})

            return Fnew, bcnew

        if order is None:
            order = self.order
        if basis_type == self.basis_type and order == self.order:
            trial_el = self.trial_el
            test_el = self.test_el
        else:
            trial_el, test_el = getElements(basis_type, order)
        quadrature = quadrature or self.quadrature
        return getFormGalerkin(F, trial_el, test_el, quadrature,
                               self.t, self.dt, u0, stages,
                               bcs=bcs, bc_type=self.bc_type,
                               aux_indices=aux_indices)

    def _update(self):
        for u0bit, expr in zip(self.u0.subfunctions, self.u_update):
            u0bit.assign(expr)

    def set_update_expressions(self):
        """Set up symbolic expressions for the update."""
        # Tabulate the trial and test basis functions at the final time
        update_trial = vecconst(self.trial_el.tabulate(0, (1.0,))[(0,)])
        update_test = vecconst(self.test_el.tabulate(0, (1.0,))[(0,)])

        i0, = self.trial_el.entity_dofs()[0][0]
        self.u_update = []
        for i in range(self.num_fields):
            ks = list(self.stages.subfunctions[i::self.num_fields])
            if self.aux_indices and i in self.aux_indices:
                wts = update_test
            else:
                wts = update_trial
                ks.insert(i0, self.u0.subfunctions[i])
            self.u_update.append(sum(w * k for w, k in zip(wts, ks)))

    def set_initial_guess(self):
        """Set a constant-in-time initial guess."""
        ref_el = self.test_el.get_reference_element()
        P0 = DiscontinuousLagrange(ref_el, 0)
        P0 = P0.get_nodal_basis()
        B = P0.get_coeffs()

        test_dual = self.test_el.get_dual_set()
        test_dofs = np.dot(test_dual.to_riesz(P0), B)

        i0, = self.trial_el.entity_dofs()[0][0]
        trial_dual = self.trial_el.get_dual_set()
        trial_dofs = np.dot(trial_dual.to_riesz(P0), B)
        trial_dofs = np.delete(trial_dofs, i0, axis=0)

        dof = Constant(0)
        for k in range(self.num_stages):
            for i, u0bit in enumerate(self.u0.subfunctions):
                sbit = self.stages.subfunctions[self.num_fields*k+i]
                if self.aux_indices and i in self.aux_indices:
                    dof.assign(test_dofs[k])
                else:
                    dof.assign(trial_dofs[k])
                if abs(float(dof)) < 1E-12:
                    sbit.zero()
                else:
                    sbit.assign(u0bit * dof)
