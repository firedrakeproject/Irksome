from functools import reduce
import FIAT
from operator import mul
from ufl.classes import Zero
from .deriv import TimeDerivative
from .stage import getBits
from .tools import MeshConstant, replace
import np
from firedrake import TestFunction, Function, NonlinearVariationalProblem as NLVP, NonlinearVariationalSolver as NLVS


def getFormGalerkin(F, L_trial, L_test, Q, t, dt, u0,
                    bcs=None, nullspace=None):

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
         containing (possible time-dependent) boundary conditions imposed
         on the system.
    :arg nullspace: A list of tuples of the form (index, VSB) where
         index is an index into the function space associated with `u`
         and VSB is a :class: `firedrake.VectorSpaceBasis` instance to
         be passed to a `firedrake.MixedVectorSpaceBasis` over the
         larger space associated with the Runge-Kutta method

    On output, we return a tuple consisting of four parts:

       - Fnew, the :class:`Form`
       - w
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the Galerkin-in-time solution,
       - 'nspnew', the :class:`firedrake.MixedVectorSpaceBasis` object
         that represents the nullspace of the coupled system
    """
    assert L_test.ref_el == FIAT.ufc_simplex(1)
    assert L_trial.ref_el == FIAT.ufc_simplex(1)
    assert Q.ref_el == FIAT.ufc_simplex(1)
    assert L_trial.order == L_test.order + 1

    v = F.arguments()[0]
    V = v.function_space()
    assert V == u0.function_space()

    num_fields = len(V)
    num_stages = L_test.space_dimension()

    Vbig = reduce(mul, (V for _ in range(num_stages)))

    VV = TestFunction(Vbig)

    UU = Function(Vbig)  # u0 + this are coefficients of the Galerkin polynomial

    qpts = Q.get_points()
    tabulate_trials = L_trial.tabulate(0, qpts)
    trial_vals = tabulate_trials[0,]
    trial_dvals = tabulate_trials[1,]
    test_vals = L_test.tabulate(0, qpts)[0,]

    if L_trial.is_nodal():
        points = []
        for ell in L_trial.dual.nodes:
            assert isinstance(ell, FIAT.functional.PointEvaluation)
            # Assert singleton point for each node.
            pt, = ell.get_point_dict().keys()
            points.append(pt[0])

        # also needed as collocation points for BC...
        c_trial = np.asarray(points)
        # GLL DOFs are ordered by increasing entity dimension!
        trial_perm = np.argsort(c_trial)
        c_trial = c_trial[trial_perm]
        trial_vals = trial_vals[trial_perm]
        trial_dvals = trial_dvals[trial_perm]
    if L_test.is_nodal():
        points = []
        for ell in L_test.dual.nodes:
            assert isinstance(ell, FIAT.functional.PointEvaluation)
            # Assert singleton point for each node.
            pt, = ell.get_point_dict().keys()
            points.append(pt[0])

        c_test = np.asarray(points)
        # GLL DOFs are ordered by increasing entity dimension!
        test_perm = np.argsort(c_test)
        c_test = c_test[test_perm]
        test_vals = test_vals[test_perm]

    u0bits, vbits, VVbits, UUbits = getBits(num_stages, num_fields,
                                            u0, UU, v, VV)

    Fnew = Zero()

    MC = MeshConstant(V.mesh())
    vecconst = np.vectorize(lambda c: MC.Constant(c))
    trial_vals = vecconst(trial_vals)
    trial_dvals = vecconst(trial_dvals)
    test_vals = vecconst(test_vals)
    qpts = vecconst(Q.get_points())
    qwts = vecconst(Q.get_weights())

    for i in range(num_stages):
        repl = {}
        for j in range(num_fields):
            repl[vbits[j]] = VVbits[i][j]
            for ii in np.ndindex(vbits[j].ufl_shape):
                repl[vbits[j][ii]] = VVbits[i][j][ii]
        F_i = replace(F, repl)

        # now loop over quadrature points
        for q in range(len(qpts)):
            repl = {t: t + dt * qpts[q]}
            for k in range(num_fields):
                tosub = u0bits[k] * trial_vals[0, q]
                d_tosub = u0bits[k] * trial_dvals[0, q]
                for ell in range(num_stages):
                    tosub += UUbits[ell][k] * trial_vals[1 + ell, q]
                    d_tosub += UUbits[ell][k] * trial_dvals[1 + ell, q]

                repl[u0bits[k]] = tosub
                repl[TimeDerivative(u0bits[k])] = d_tosub / dt

                for ii in np.ndindex(u0bits[k].ufl_shape):
                    tosub = u0bits[k][ii] * trial_vals[0, q]
                    d_tosub = u0bits[k][ii] * trial_dvals[0, q]
                    for ell in range(num_stages):
                        tosub += UUbits[ell][k][ii] * trial_vals[1 + ell, q]
                        d_tosub += UUbits[ell][k][ii] * trial_dvals[1 + ell, q]
                    repl[u0bits[k][ii]] = tosub
                    repl[TimeDerivative(u0bits[k][ii])] = d_tosub / dt
            Fnew += dt * qwts[q] * test_vals[i, q] * replace(F_i, repl)


# class GalkerinTimeStepper:
#     """Front-end class for advancing a time-dependent PDE via a Galerkin
#     in time method

#     :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
#             F(t, u; v) == 0, where `u` is the unknown
#             :class:`firedrake.Function and `v` is the
#             :class:firedrake.TestFunction`.
#     :arg V_trial: A :class:`firedrake.FunctionSpace` for the trial functions in time
#     :arg V_test: A :class:`firedrake.FunctionSpace` for the test functions in time
#     :arg quad: A :class:`fiat.QuadratureRule` for the time integration
#     :arg t: a :class:`Function` on the Real space over the same mesh as
#          `u0`.  This serves as a variable referring to the current time.
#     :arg dt: a :class:`Function` on the Real space over the same mesh as
#          `u0`.  This serves as a variable referring to the current time step.
#          The user may adjust this value between time steps.
#     :arg u0: A :class:`firedrake.Function` containing the current
#             state of the problem to be solved.
#     :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing
#             the strongly-enforced boundary conditions.  Irksome will
#             manipulate these to obtain boundary conditions for each
#             stage of the method.
#     :arg solver_parameters: A :class:`dict` of solver parameters that
#             will be used in solving the algebraic problem associated
#             with each time step.
#     :arg appctx: An optional :class:`dict` containing application context.
#             This gets included with particular things that Irksome will
#             pass into the nonlinear solver so that, say, user-defined preconditioners
#             have access to it.
#     :arg nullspace: A list of tuples of the form (index, VSB) where
#             index is an index into the function space associated with
#             `u` and VSB is a :class: `firedrake.VectorSpaceBasis`
#             instance to be passed to a
#             `firedrake.MixedVectorSpaceBasis` over the larger space
#             associated with the Runge-Kutta method
#     """
#     def __init__(self, F, V_trial, V_test, quad, t, dt, u0, bcs=None,
#                  solver_parameters=None, appctx=None, nullspace=None):
#         self.u0 = u0
#         self.F = F
#         self.orig_bcs = bcs
#         self.t = t
#         self.dt = dt
#         self.V_trial = V_trial
#         self.V_test = V_test
#         self.quad = quad
#         self.num_steps = 0
#         self.num_nonlinear_iterations = 0
#         self.num_linear_iterations = 0

#         bigF, bigBasis, bigBCs, bigNSP = \
#             getFormGalerkin(F, V_trial, V_test, quad, t, dt, u0, bcs, nullspace)

#         self.bigBasis = bigBasis
#         self.bigBCs = bigBCs
#         problem = NLVP(bigF, bigBasis, bigBCs)
#         appctx_irksome = {"F": F,
#                           "t": t,
#                           "dt": dt,
#                           "u0": u0,
#                           "bcs": bcs,
#                           "nullspace": nullspace}
#         if appctx is None:
#             appctx = appctx_irksome
#         else:
#             appctx = {**appctx, **appctx_irksome}

#         push_parent(u0.function_space().dm, stages.function_space().dm)
#         self.solver = NLVS(problem,
#                            appctx=appctx,
#                            solver_parameters=solver_parameters,
#                            nullspace=bigNSP)
#         pop_parent(u0.function_space().dm, stages.function_space().dm)

#     def _update(self):
#         """Assuming the algebraic problem for the Galerkin problems has been
#         solved, updates the solution.  This will not typically be
#         called by an end user."""

#         # ws = self.bigBasis
#         # u0bits = u0.subfunctions
#         # for s in range(ns):
#         #     for i, u0bit in enumerate(u0bits):
#         #         u0bit = 

#     def advance(self):
#         """Advances the system from time `t` to time `t + dt`.
#         Note: overwrites the value `u0`."""
#         push_parent(self.u0.function_space().dm, self.stages.function_space().dm)
#         self.solver.solve()
#         pop_parent(self.u0.function_space().dm, self.stages.function_space().dm)

#         self.num_steps += 1
#         self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
#         self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()
#         self._update()

#     def solver_stats(self):
#         return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)