from functools import reduce
from FIAT import (Bernstein, DiscontinuousElement, DiscontinuousLagrange,
                  Lagrange, make_quadrature, ufc_simplex)
from FIAT.functional import PointEvaluation
from operator import mul
from ufl.classes import Zero
from ufl.constantvalue import as_ufl
from .bcs import stage2spaces4bc
from .manipulation import extract_terms, strip_dt_form
from .stage import getBits
from .tools import MeshConstant, getNullspace, replace
import numpy as np
from firedrake import TestFunction, Function, NonlinearVariationalProblem as NLVP, NonlinearVariationalSolver as NLVS
from firedrake.dmhooks import pop_parent, push_parent


def getFormDiscGalerkin(F, L, Q, t, dt, u0, bcs=None, nullspace=None):

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
       - UU, the :class:`Function` representing the stages to be solved for
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the Galerkin-in-time solution,
       - 'nspnew', the :class:`firedrake.MixedVectorSpaceBasis` object
         that represents the nullspace of the coupled system
    """
    assert L.get_reference_element() == ufc_simplex(1)
    assert Q.ref_el == ufc_simplex(1)

    v = F.arguments()[0]
    V = v.function_space()
    assert V == u0.function_space()

    MC = MeshConstant(V.mesh())
    vecconst = np.vectorize(lambda c: MC.Constant(c))

    num_fields = len(V)
    num_stages = L.space_dimension()

    Vbig = reduce(mul, (V for _ in range(num_stages)))

    VV = TestFunction(Vbig)
    UU = Function(Vbig)

    qpts = Q.get_points()
    qwts = Q.get_weights()

    tabulate_basis = L.tabulate(1, qpts)
    basis_vals = tabulate_basis[0,]
    basis_dvals = tabulate_basis[1,]

    is_bernstein = False
    if isinstance(L, Bernstein):
        is_bernstein = True
    elif isinstance(L, DiscontinuousElement):
        if isinstance(L._element, Bernstein):
            is_bernstein = True

    if not is_bernstein:
        points = []
        for ell in L.dual.nodes:
            assert isinstance(ell, PointEvaluation)
            # Assert singleton point for each node.
            pt, = ell.get_point_dict().keys()
            points.append(pt[0])

        # also needed as collocation points for BC...
        c = np.asarray(points)
        # GLL DOFs are ordered by increasing entity dimension!
        perm = np.argsort(c)
        c = c[perm]
        basis_vals = basis_vals[perm]
        basis_dvals = basis_dvals[perm]

    # mass matrix later for BC
    mmat = basis_vals @ np.diag(qwts) @ basis_vals.T
    mmat_inv = vecconst(np.linalg.inv(mmat))

    u0bits, vbits, VVbits, UUbits = getBits(num_stages, num_fields,
                                            u0, UU, v, VV)

    split_form = extract_terms(F)
    dtless = strip_dt_form(split_form.time)

    Fnew = Zero()

    basis_vals = vecconst(basis_vals)
    basis_dvals = vecconst(basis_dvals)

    qpts = vecconst(qpts.reshape((-1,)))
    qwts = vecconst(qwts)

    # Terms with time derivatives
    for i in range(num_stages):
        repl = {}
        for j in range(num_fields):
            repl[vbits[j]] = VVbits[i][j]
            for ii in np.ndindex(vbits[j].ufl_shape):
                repl[vbits[j][ii]] = VVbits[i][j][ii]
        F_i = replace(dtless, repl)

        # now loop over quadrature points
        for q in range(len(qpts)):
            repl = {t: t + dt * qpts[q]}
            for k in range(num_fields):
                d_tosub = sum(basis_dvals[ell, q] * UUbits[ell][k] for ell in range(num_stages)) / dt
                repl[u0bits[k]] = d_tosub

                for ii in np.ndindex(u0bits[k].ufl_shape):
                    d_tosub = sum(basis_dvals[ell, q] * UUbits[ell][k][ii]
                                  for ell in range(num_stages)) / dt
                    repl[u0bits[k][ii]] = d_tosub
            Fnew += dt * qwts[q] * basis_vals[i, q] * replace(F_i, repl)

    # jump terms
    repl = {}
    for k in range(num_fields):
        repl[u0bits[k]] = UUbits[0][k] - u0bits[k]
        repl[vbits[k]] = VVbits[0][k]
        for ii in np.ndindex(u0bits[k].ufl_shape):
            repl[u0bits[k][ii]] = UUbits[0][k][ii] - u0bits[k][ii]
            repl[vbits[k][ii]] = VVbits[0][k][ii]
    Fnew += replace(dtless, repl)

    # handle the rest of the terms
    for i in range(num_stages):
        repl = {}
        for j in range(num_fields):
            repl[vbits[j]] = VVbits[i][j]
            for ii in np.ndindex(vbits[j].ufl_shape):
                repl[vbits[j][ii]] = VVbits[i][j][ii]
        F_i = replace(split_form.remainder, repl)

        # now loop over quadrature points
        for q in range(len(qpts)):
            repl = {t: t + dt * qpts[q]}
            for k in range(num_fields):
                tosub = sum(basis_vals[ell, q] * UUbits[ell][k] for ell in range(num_stages))
                repl[u0bits[k]] = tosub

                for ii in np.ndindex(u0bits[k].ufl_shape):
                    tosub = sum(basis_vals[ell, q] * UUbits[ell][k][ii]
                                for ell in range(num_stages))
                    repl[u0bits[k][ii]] = tosub
            Fnew += dt * qwts[q] * basis_vals[i, q] * replace(F_i, repl)

    # Oh, honey, is it the boundary conditions?
    if bcs is None:
        bcs = []
    bcsnew = []
    for bc in bcs:
        bcarg = as_ufl(bc._original_arg)
        bcblah_at_qp = np.zeros((len(qpts),), dtype="O")
        for q in range(len(qpts)):
            bcblah_at_qp[q] = qwts[q] * (
                replace(bcarg, {t: t + qpts[q] * dt}))
        bc_func_for_stages = mmat_inv @ (basis_vals @ bcblah_at_qp)
        for i in range(num_stages):
            Vbigi = stage2spaces4bc(bc, V, Vbig, i)
            bcsnew.append(bc.reconstruct(V=Vbigi, g=bc_func_for_stages[i]))

    return Fnew, UU, bcsnew, getNullspace(V, Vbig, num_stages, nullspace)


class DiscontinuousGalerkinTimeStepper:
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
    :arg basis_type: A string indicating if the standard (discontinuous)
            Lagrange basis is to be used for the time stages or if the Bernstein
            basis is to be used instead
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
                 quadrature=None,
                 solver_parameters=None, appctx=None, nullspace=None):
        assert order >= 0
        self.u0 = u0
        self.orig_bcs = bcs
        self.t = t
        self.dt = dt
        self.order = order
        if basis_type is None:
            basis_type = "Lagrange"
        self.basis_type = basis_type

        V = u0.function_space()
        self.num_fields = len(V)

        ufc_line = ufc_simplex(1)

        if order == 0:
            self.el = DiscontinuousLagrange(ufc_line, 0)            
        elif basis_type == "Lagrange":
            self.el = DiscontinuousElement(Lagrange(ufc_line, order))
        elif basis_type == "Bernstein":
            self.el = DiscontinuousElement(Bernstein(ufc_line, order))
        else:
            raise NotImplementedError("Not implemented basis type")

        if quadrature is None:
            quadrature = make_quadrature(ufc_line, order+1)
        self.quadrature = quadrature
        assert np.size(quadrature.get_points()) >= order+1

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        bigF, UU, bigBCs, bigNSP = \
            getFormDiscGalerkin(F, self.el,
                                quadrature, t, dt, u0, bcs, nullspace)

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
            u0bit.assign(UUs[self.num_fields*(self.order)+i])

    def solver_stats(self):
        return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)
