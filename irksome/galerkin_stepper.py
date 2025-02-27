from functools import reduce
from FIAT import (Bernstein, DiscontinuousElement, DiscontinuousLagrange,
                  IntegratedLegendre, Lagrange, Legendre,
                  make_quadrature, ufc_simplex)
from operator import mul
from ufl import zero
from ufl.constantvalue import as_ufl
from .bcs import bc2space, stage2spaces4bc
from .deriv import TimeDerivative
from .tools import getNullspace, ConstantOrZero, component_replace, replace
import numpy as np
from firedrake import as_vector, dot, Constant, TestFunction, Function, NonlinearVariationalProblem as NLVP, NonlinearVariationalSolver as NLVS
from firedrake.dmhooks import pop_parent, push_parent


def getFormGalerkin(F, L_trial, L_test, Q, t, dt, u0, bcs=None):

    """Given a time-dependent variational form, trial and test spaces, and
    a quadrature rule, produce UFL for the Galerkin-in-Time method.

    :arg F: UFL form for the semidiscrete ODE/DAE
    :arg L_trial: A :class:`FIAT.FiniteElement` for the trial functions in time
    :arg L_test: A :class:`FIAT.FinteElement` for the test functions in time
    :arg Q: A :class:`FIAT.QuadratureRule` for the time integration
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
         containing (possibly time-dependent) boundary conditions imposed
         on the system.

    On output, we return a tuple consisting of four parts:

       - Fnew, the :class:`Form` corresponding to the Galerkin-in-Time discretized problem
       - UU, the :class:`Function` representing the stages to be solved for
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the Galerkin-in-time solution,
       - 'nspnew', the :class:`firedrake.MixedVectorSpaceBasis` object
         that represents the nullspace of the coupled system
    """
    assert L_test.get_reference_element() == Q.ref_el
    assert L_trial.get_reference_element() == Q.ref_el
    assert Q.ref_el.get_spatial_dimension() == 1
    assert L_trial.get_order() == L_test.get_order() + 1

    v = F.arguments()[0]
    V = v.function_space()
    assert V == u0.function_space()

    num_stages = L_test.space_dimension()

    Vbig = reduce(mul, (V for _ in range(num_stages)))
    VV = TestFunction(Vbig)
    UU = Function(Vbig)  # u0 + this are coefficients of the Galerkin polynomial

    u_np = np.reshape(UU, (num_stages, *u0.ufl_shape))
    v_np = np.reshape(VV, (num_stages, *u0.ufl_shape))

    qpts = Q.get_points()
    qwts = Q.get_weights()

    tabulate_trials = L_trial.tabulate(1, qpts)
    trial_vals = tabulate_trials[(0,)]
    trial_dvals = tabulate_trials[(1,)]
    test_vals = L_test.tabulate(0, qpts)[(0,)]

    # sort dofs geometrically by entity location
    edofs = L_trial.entity_dofs()
    trial_perm = [*edofs[0][0], *edofs[1][0], *edofs[0][1]]
    trial_vals = trial_vals[trial_perm]
    trial_dvals = trial_dvals[trial_perm]

    # mass-ish matrix later for BC
    mmat = np.multiply(test_vals, qwts) @ trial_vals[1:].T

    # L2 projector
    proj = Constant(np.linalg.solve(mmat, np.multiply(test_vals, qwts)))

    Fnew = zero()

    vecconst = np.vectorize(ConstantOrZero)
    trial_vals = vecconst(trial_vals)
    trial_dvals = vecconst(trial_dvals)
    test_vals = vecconst(test_vals)
    qpts = vecconst(qpts.reshape((-1,)))
    qwts = vecconst(qwts)

    for i in range(num_stages):
        repl = {v: v_np[i]}
        F_i = component_replace(F, repl)

        # now loop over quadrature points
        for q in range(len(qpts)):
            tosub = u0 * trial_vals[0, q]
            tosub += sum(u_np[j] * trial_vals[1+j, q] for j in range(num_stages))

            d_tosub = u0 * trial_dvals[0, q]
            d_tosub += sum(u_np[j] * trial_dvals[1+j, q] for j in range(num_stages))

            repl = {t: t + dt * qpts[q],
                    u0: tosub,
                    TimeDerivative(u0): d_tosub / dt}

            Fnew += dt * qwts[q] * test_vals[i, q] * component_replace(F_i, repl)

    # Oh, honey, is it the boundary conditions?
    if bcs is None:
        bcs = []
    bcsnew = []
    for bc in bcs:
        u0_sub = bc2space(bc, u0)
        bcarg = as_ufl(bc._original_arg)
        bcblah_at_qp = np.zeros((len(qpts),), dtype="O")
        for q in range(len(qpts)):
            tcur = t + qpts[q] * dt
            bcblah_at_qp[q] = replace(bcarg, {t: tcur}) - u0_sub * trial_vals[0, q]
        bc_func_for_stages = dot(proj, as_vector(bcblah_at_qp))
        for i in range(num_stages):
            Vbigi = stage2spaces4bc(bc, V, Vbig, i)
            bcsnew.append(bc.reconstruct(V=Vbigi, g=bc_func_for_stages[i]))

    return Fnew, UU, bcsnew


class GalerkinTimeStepper:
    """Front-end class for advancing a time-dependent PDE via a Galerkin
    in time method

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg order: an integer indicating the order of the DG space to use
         (with order == 1 corresponding to CG(1)-in-time for the trial space)
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
            to be used in time, defaulting to GL with order points
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
                 quadrature=None,
                 solver_parameters=None, appctx=None, nullspace=None):
        assert order >= 1
        self.u0 = u0
        self.orig_bcs = bcs
        self.t = t
        self.dt = dt
        self.order = order
        self.basis_type = basis_type

        V = u0.function_space()
        self.num_fields = len(V)

        ufc_line = ufc_simplex(1)
        if basis_type == "Bernstein":
            self.trial_el = Bernstein(ufc_line, order)
            if order == 1:
                self.test_el = DiscontinuousLagrange(ufc_line, 0)
            else:
                self.test_el = DiscontinuousElement(
                    Bernstein(ufc_line, order-1))
        elif basis_type == "integral":
            self.trial_el = IntegratedLegendre(ufc_line, order)
            self.test_el = Legendre(ufc_line, order-1)
        else:
            # Let recursivenodes handle the general case
            variant = None if basis_type == "Lagrange" else basis_type
            self.trial_el = Lagrange(ufc_line, order, variant=variant)
            self.test_el = DiscontinuousLagrange(ufc_line, order-1, variant=variant)

        if quadrature is None:
            quadrature = make_quadrature(ufc_line, order)
        self.quadrature = quadrature
        assert np.size(quadrature.get_points()) >= order

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        bigF, UU, bigBCs = \
            getFormGalerkin(F, self.trial_el, self.test_el,
                            quadrature, t, dt, u0, bcs)

        bigNSP = getNullspace(V, UU.function_space(),
                              order, nullspace)
        
        self.UU = UU
        self.bigBCs = bigBCs
        problem = NLVP(bigF, UU, bigBCs)
        appctx_irksome = {"F": F,
                          "t": t,
                          "dt": dt,
                          "u0": u0,
                          "bcs": bcs,
                          "nullspace": nullspace}
        if appctx is None:
            appctx = appctx_irksome
        else:
            appctx = {**appctx, **appctx_irksome}

        push_parent(u0.function_space().dm, UU.function_space().dm)
        self.solver = NLVS(problem,
                           appctx=appctx,
                           solver_parameters=solver_parameters,
                           nullspace=bigNSP)
        pop_parent(u0.function_space().dm, UU.function_space().dm)

    def advance(self):
        """Advances the system from time `t` to time `t + dt`.
        Note: overwrites the value `u0`."""
        push_parent(self.u0.function_space().dm, self.UU.function_space().dm)
        self.solver.solve()
        pop_parent(self.u0.function_space().dm, self.UU.function_space().dm)

        self.num_steps += 1
        self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()

        u0 = self.u0
        u0bits = u0.subfunctions
        UUs = self.UU.subfunctions

        for i, u0bit in enumerate(u0bits):
            u0bit.assign(UUs[self.num_fields*(self.order-1)+i])

    def solver_stats(self):
        return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)
