import FIAT
import numpy as np
from firedrake import (Function, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, TestFunction,
                       as_ufl, dx, inner)
from firedrake.dmhooks import pop_parent, push_parent
from ufl import zero

from .ButcherTableaux import RadauIIA
from .deriv import TimeDerivative
from .stage_value import getFormStage
from .tools import AI, ConstantOrZero, IA, MeshConstant, replace, component_replace, getNullspace, get_stage_space
from .bcs import bc2space


def riia_explicit_coeffs(k):
    """Computes the coefficients needed for the explicit part
    of a RadauIIA-IMEX method."""
    U = FIAT.ufc_simplex(1)
    L = FIAT.GaussRadau(U, k - 1)

    Q = FIAT.make_quadrature(L.ref_el, 2*k)
    qpts = Q.get_points()
    qwts = Q.get_weights()

    A = np.zeros((k, k))
    for i, ell in enumerate(L.dual.nodes):
        pt, = ell.pt_dict
        ci, = pt
        qpts_i = 1 + qpts * ci
        qwts_i = qwts * ci
        Lvals_i = L.tabulate(0, qpts_i)[(0,)]
        A[i, :] = Lvals_i @ qwts_i

    return A


def getFormExplicit(Fexp, butch, u0, UU, t, dt, splitting=None):
    """Processes the explicitly split-off part for a RadauIIA-IMEX
    method.  Returns the forms for both the iterator and propagator,
    which really just differ by which constants are in them."""
    v = Fexp.arguments()[0]
    Vbig = UU.function_space()
    VV = TestFunction(Vbig)

    num_stages = butch.num_stages

    Aexp = riia_explicit_coeffs(num_stages)

    vecconst = np.vectorize(ConstantOrZero)

    Aprop = vecconst(Aexp)
    Ait = vecconst(butch.A)
    C = vecconst(butch.c)

    v_np = np.reshape(VV, (num_stages, *u0.ufl_shape))
    u_np = np.reshape(UU, (num_stages, *u0.ufl_shape))

    Fit = zero()
    Fprop = zero()

    if splitting == AI:
        for i in range(num_stages):
            # replace test function
            repl = {v: v_np[i]}
            Ftmp = component_replace(Fexp, repl)

            # replace the solution with stage values
            for j in range(num_stages):
                repl = {t: t + C[j] * dt,
                        u0: u_np[j]}

                # and sum the contribution
                replF = component_replace(Ftmp, repl)
                Fit += Ait[i, j] * dt * replF
                Fprop += Aprop[i, j] * dt * replF
    elif splitting == IA:
        # diagonal contribution to iterator
        for i in range(num_stages):
            repl = {t: t+C[i]*dt,
                    u0: u_np[i],
                    v: v_np[i]}

            Fit += dt * component_replace(Fexp, repl)

        # dense contribution to propagator
        AinvAexp = vecconst(np.linalg.solve(butch.A, Aexp))

        for i in range(num_stages):
            # replace test function
            repl = {v: v_np[i]}
            Ftmp = component_replace(Fexp, repl)

            # replace the solution with stage values
            for j in range(num_stages):
                repl = {t: t + C[j] * dt,
                        u0: u_np[j]}

                # and sum the contribution
                Fprop += AinvAexp[i, j] * dt * component_replace(Ftmp, repl)
    else:
        raise NotImplementedError(
            "Must specify splitting to either IA or AI")

    return Fit, Fprop


class RadauIIAIMEXMethod:
    """Class for advancing a time-dependent PDE via a polynomial
    IMEX/RadauIIA method.  This requires one to split the PDE into
    an implicit and explicit part.
    The class sets up two methods -- `advance` and `iterate`.
    The former is used to move the solution forward in time,
    while the latter is used both to start the method (filling up
    the initial stage values) and can be used at each time step
    to increase the accuracy/stability.  In the limit as
    the iterator is applied many times per time step,
    one expects convergence to the solution that would have been
    obtained from fully-implicit RadauIIA method.

    :arg F: A :class:`ufl.Form` instance describing the implicit part
            of the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg Fexp: A :class:`ufl.Form` instance describing the part of the
            PDE that is explicitly split off.
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
            the Runge-Kutta method to be used for time marching.
            Only RadauIIA is allowed here (but it can be any number of stages).
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
            stage of the RK method.
    :arg it_solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with the iterator.
    :arg prop_solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with the propagator.
    :arg splitting: A callable used to factor the Butcher matrix,
            currently, only AI is supported.
    :arg appctx: An optional :class:`dict` containing application context.
    :arg nullspace: An optional null space object.
    """
    def __init__(self, F, Fexp, butcher_tableau,
                 t, dt, u0, bcs=None,
                 it_solver_parameters=None,
                 prop_solver_parameters=None,
                 splitting=AI,
                 appctx=None,
                 nullspace=None,
                 num_its_initial=0,
                 num_its_per_step=0):
        assert isinstance(butcher_tableau, RadauIIA)

        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = len(butcher_tableau.b)
        self.butcher_tableau = butcher_tableau
        self.num_its_initial = num_its_initial
        self.num_its_per_step = num_its_per_step

        # solver statistics
        self.num_steps = 0
        self.num_props = 0
        self.num_its = 0
        self.num_nonlinear_iterations_prop = 0
        self.num_nonlinear_iterations_it = 0
        self.num_linear_iterations_prop = 0
        self.num_linear_iterations_it = 0

        # Since this assumes stiff accuracy, we drop
        # the update information on the floor.
        V = u0.function_space()
        Vbig = get_stage_space(V, self.num_stages)
        UU = Function(Vbig)

        Fbig, bigBCs = getFormStage(
            F, butcher_tableau, t, dt, u0, UU, bcs,
            splitting=splitting)

        nsp = getNullspace(u0.function_space(),
                           UU.function_space(),
                           self.num_stages, nullspace)

        self.UU = UU
        self.UU_old = UU_old = Function(UU.function_space())
        self.UU_old_split = UU_old.subfunctions
        self.bigBCs = bigBCs

        Fit, Fprop = getFormExplicit(
            Fexp, butcher_tableau, u0, UU_old, t, dt, splitting)

        self.itprob = NonlinearVariationalProblem(
            Fbig + Fit, UU, bcs=bigBCs)
        self.propprob = NonlinearVariationalProblem(
            Fbig + Fprop, UU, bcs=bigBCs)

        appctx_irksome = {"F": F,
                          "Fexp": Fexp,
                          "butcher_tableau": butcher_tableau,
                          "t": t,
                          "dt": dt,
                          "u0": u0,
                          "bcs": bcs,
                          "stage_type": "value",
                          "splitting": splitting,
                          "nullspace": nullspace}
        if appctx is None:
            appctx = appctx_irksome
        else:
            appctx = {**appctx, **appctx_irksome}

        push_parent(self.u0.function_space().dm, self.UU.function_space().dm)
        self.it_solver = NonlinearVariationalSolver(
            self.itprob, appctx=appctx,
            solver_parameters=it_solver_parameters,
            nullspace=nsp)
        self.prop_solver = NonlinearVariationalSolver(
            self.propprob, appctx=appctx,
            solver_parameters=prop_solver_parameters,
            nullspace=nsp)
        pop_parent(self.u0.function_space().dm, self.UU.function_space().dm)

        num_fields = len(self.u0.function_space())
        u0split = u0.subfunctions
        for i, u0bit in enumerate(u0split):
            for s in range(self.num_stages):
                ii = s * num_fields + i
                self.UU_old_split[ii].assign(u0bit)

        for _ in range(num_its_initial):
            self.iterate()

    def iterate(self):
        """Called 1 or more times to set up the initial state of the
        system before time-stepping.  Can also be called after each
        call to `advance`"""
        push_parent(self.u0.function_space().dm, self.UU.function_space().dm)
        self.it_solver.solve()
        pop_parent(self.u0.function_space().dm, self.UU.function_space().dm)
        self.UU_old.assign(self.UU)
        self.num_its += 1
        self.num_nonlinear_iterations_it += self.it_solver.snes.getIterationNumber()
        self.num_linear_iterations_it += self.it_solver.snes.getLinearSolveIterations()

    def propagate(self):
        """Moves the solution forward in time, to be followed by 0 or
        more calls to `iterate`."""

        ns = self.num_stages
        nf = self.num_fields
        u0split = self.u0.subfunctions
        for i, u0bit in enumerate(u0split):
            u0bit.assign(self.UU_old_split[(ns-1)*nf + i])

        push_parent(self.u0.function_space().dm, self.UU.function_space().dm)

        ps = self.prop_solver
        ps.solve()
        pop_parent(self.u0.function_space().dm, self.UU.function_space().dm)
        self.UU_old.assign(self.UU)
        self.num_props += 1
        self.num_nonlinear_iterations_prop += ps.snes.getIterationNumber()
        self.num_linear_iterations_prop += ps.snes.getLinearSolveIterations()

    def advance(self):
        self.propagate()
        for _ in range(self.num_its_per_step):
            self.iterate()
        self.num_steps += 1

    def solver_stats(self):
        return (self.num_steps, self.num_props, self.num_its,
                self.num_nonlinear_iterations_prop,
                self.num_linear_iterations_prop,
                self.num_nonlinear_iterations_it,
                self.num_linear_iterations_it)


def getFormsDIRKIMEX(F, Fexp, ks, khats, butch, t, dt, u0, bcs=None):
    if bcs is None:
        bcs = []

    v = F.arguments()[0]
    V = v.function_space()
    msh = V.mesh()
    assert V == u0.function_space()

    num_stages = butch.num_stages
    k = Function(V)
    g = Function(V)

    khat = Function(V)
    ghat = Function(V)
    vhat = TestFunction(V)

    # Note: the Constant c is used for substitution in both the
    # implicit variational form and BC's, and we update it for each stage in
    # the loop over stages in the advance method.  The Constants a and chat are
    # used similarly in the variational forms
    MC = MeshConstant(msh)
    c = MC.Constant(1.0)
    chat = MC.Constant(1.0)
    a = MC.Constant(1.0)

    # Implicit replacement, solve at time t + c * dt, for k
    repl = {t: t + c * dt,
            u0: g + dt * a * k,
            TimeDerivative(u0): k}
    stage_F = component_replace(F, repl)

    # Explicit replacement, solve at time t + chat * dt, for khat
    replhat = {t: t + chat * dt,
               u0: ghat}

    Fhat = inner(khat, vhat)*dx + component_replace(Fexp, replhat)

    bcnew = []

    # For the DIRK-IMEX case, we need one new BC for each old one
    # (rather than one per stage), but we need a `Function` inside of
    # each BC and a rule for computing that function at each time for
    # each stage.

    a_vals = np.array([MC.Constant(0) for i in range(num_stages)],
                      dtype=object)
    ahat_vals = np.array([MC.Constant(0) for i in range(num_stages)],
                         dtype=object)
    d_val = MC.Constant(1.0)

    for bc in bcs:
        bcarg = as_ufl(bc._original_arg)
        bcarg_stage = replace(bcarg, {t: t+c*dt})

        gdat = bcarg_stage - bc2space(bc, u0)
        for i in range(num_stages):
            gdat -= dt*(a_vals[i]*bc2space(bc, ks[i]) + ahat_vals[i]*bc2space(bc, khats[i]))

        gdat /= dt*d_val
        bcnew.append(bc.reconstruct(g=gdat))

    return stage_F, (k, g, a, c), bcnew, Fhat, (khat, ghat, chat), (a_vals, ahat_vals, d_val)


class DIRKIMEXMethod:
    """Front-end class for advancing a time-dependent PDE via a
    diagonally-implicit Runge-Kutta IMEX method formulated in terms of
    stage derivatives.  This implementation assumes a weak form
    written as F + F_explicit = 0, where both F and F_explicit are UFL
    Forms, with terms in F to be handled implicitly and those in
    F_explicit to be handled explicitly
    """

    def __init__(self, F, F_explicit, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None, mass_parameters=None, appctx=None, nullspace=None):
        assert butcher_tableau.is_dirk_imex

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0
        self.num_mass_nonlinear_iterations = 0
        self.num_mass_linear_iterations = 0

        self.butcher_tableau = butcher_tableau
        self.num_stages = butcher_tableau.num_stages

        self.V = V = u0.function_space()
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.ks = [Function(V) for _ in range(self.num_stages)]
        self.k_hat_s = [Function(V) for _ in range(self.num_stages)]

        stage_F, (k, g, a, c), bcnew, Fhat, (khat, ghat, chat), (a_vals, ahat_vals, d_val) = getFormsDIRKIMEX(
            F, F_explicit, self.ks, self.k_hat_s, butcher_tableau, t, dt, u0, bcs=bcs)

        self.bcnew = bcnew

        appctx_irksome = {"F": F,
                          "F_explicit": F_explicit,
                          "butcher_tableau": butcher_tableau,
                          "t": t,
                          "dt": dt,
                          "u0": u0,
                          "bcs": bcs,
                          "bc_type": "DAE",
                          "nullspace": nullspace}
        if appctx is None:
            appctx = appctx_irksome
        else:
            appctx = {**appctx, **appctx_irksome}

        self.problem = NonlinearVariationalProblem(stage_F, k, bcnew)
        self.solver = NonlinearVariationalSolver(self.problem, appctx=appctx,
                                                 solver_parameters=solver_parameters,
                                                 nullspace=nullspace)

        self.mass_problem = NonlinearVariationalProblem(Fhat, khat)
        self.mass_solver = NonlinearVariationalSolver(self.mass_problem,
                                                      solver_parameters=mass_parameters)

        self.kgac = k, g, a, c
        self.kgchat = khat, ghat, chat
        self.bc_constants = a_vals, ahat_vals, d_val

        AA = butcher_tableau.A
        A_hat = butcher_tableau.A_hat
        BB = butcher_tableau.b
        B_hat = butcher_tableau.b_hat

        if np.abs(AA[0, 0]) <= 1e-15:
            self._initialize = self._initialize_explicit
        else:
            self._initialize = self._initialize_implicit

        if B_hat[-1] == 0:
            if np.allclose(AA[-1, :], BB) and np.allclose(A_hat[-1, :], B_hat):
                self._finalize = self._finalize_stiffly_accurate
            else:
                self._finalize = self._finalize_no_last_explicit
        else:
            self._finalize = self._finalize_general

    def advance(self):
        k, g, a, c = self.kgac
        khat, ghat, chat = self.kgchat
        ks = self.ks
        k_hat_s = self.k_hat_s
        u0 = self.u0
        dtc = float(self.dt)
        bt = self.butcher_tableau
        ns = self.num_stages
        AA = bt.A
        A_hat = bt.A_hat
        CC = bt.c
        C_hat = bt.c_hat
        a_vals, ahat_vals, d_val = self.bc_constants

        # Calculating the first stage outside the loop allows boundary conditions to be enforced at
        # the end of each stage
        self._initialize()

        for i in range(1, ns):

            # Solve explicit part for previous iteration
            chat.assign(C_hat[i-1])
            self.mass_solver.solve()
            self.num_mass_nonlinear_iterations += self.mass_solver.snes.getIterationNumber()
            self.num_mass_linear_iterations += self.mass_solver.snes.getLinearSolveIterations()
            k_hat_s[i-1].assign(khat)

            g.assign(u0)
            # Update g with contributions from previous stages
            for j in range(i):
                ksplit = ks[j].subfunctions
                k_hat_split = k_hat_s[j].subfunctions
                for gbit, kbit, k_hat_bit in zip(g.subfunctions, ksplit, k_hat_split):
                    gbit += dtc * (float(AA[i, j]) * kbit + float(A_hat[i, j]) * k_hat_bit)

            # Solve for current stage
            for j in range(i):
                a_vals[j].assign(AA[i, j])
                ahat_vals[j].assign(A_hat[i, j])
            for j in range(i, ns):
                a_vals[j].assign(0)
                ahat_vals[j].assign(0)
            d_val.assign(AA[i, i])

            # Solve the nonlinear problem at stage i
            a.assign(AA[i, i])
            c.assign(CC[i])
            self.solver.solve()
            self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
            self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()
            ks[i].assign(k)

            # Update the solution for next stage
            for ghatbit, gbit, kbit in zip(ghat.subfunctions, g.subfunctions, ks[i].subfunctions):
                ghatbit.assign(gbit)
                ghatbit += dtc * float(AA[i, i]) * kbit

        self._finalize()
        self.num_steps += 1

    # No implicit first step, like in ARS schemes
    def _initialize_explicit(self):
        khat, ghat, chat = self.kgchat
        u0 = self.u0
        ghat.assign(u0)

    # Implicit first stage in general case
    def _initialize_implicit(self):
        k, g, a, c = self.kgac
        khat, ghat, chat = self.kgchat
        ks = self.ks
        u0 = self.u0
        dtc = float(self.dt)
        bt = self.butcher_tableau
        ns = self.num_stages
        AA = bt.A
        CC = bt.c
        a_vals, ahat_vals, d_val = self.bc_constants

        g.assign(u0)

        # Solve for first stage
        for j in range(ns):
            a_vals[j].assign(0)
            ahat_vals[j].assign(0)
        d_val.assign(AA[0, 0])

        a.assign(AA[0, 0])
        c.assign(CC[0])
        self.solver.solve()
        self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()
        ks[0].assign(k)

        # Update the solution for second stage
        for ghatbit, gbit, kbit in zip(ghat.subfunctions, g.subfunctions, ks[0].subfunctions):
            ghatbit.assign(gbit)
            ghatbit += dtc * float(AA[0, 0]) * kbit

    # Last part of advance for the general case, where last explicit stage is calculated and used
    def _finalize_general(self):
        khat, ghat, chat = self.kgchat
        ks = self.ks
        k_hat_s = self.k_hat_s
        u0 = self.u0
        dtc = float(self.dt)
        bt = self.butcher_tableau
        ns = self.num_stages
        C_hat = bt.c_hat
        BB = bt.b
        B_hat = bt.b_hat

        chat.assign(C_hat[ns-1])
        self.mass_solver.solve()
        self.num_mass_nonlinear_iterations += self.mass_solver.snes.getIterationNumber()
        self.num_mass_linear_iterations += self.mass_solver.snes.getLinearSolveIterations()
        k_hat_s[ns-1].assign(khat)

        # Final solution update
        for i in range(ns):
            for u0bit, kbit, k_hat_bit in zip(u0.subfunctions, ks[i].subfunctions,
                                              k_hat_s[i].subfunctions):
                u0bit += dtc * (float(BB[i]) * kbit + float(B_hat[i]) * k_hat_bit)

    # Last part of advance for the case where last explicit stage is not used
    def _finalize_no_last_explicit(self):
        ks = self.ks
        k_hat_s = self.k_hat_s
        u0 = self.u0
        dtc = float(self.dt)
        bt = self.butcher_tableau
        ns = self.num_stages
        BB = bt.b
        B_hat = bt.b_hat

        # Final solution update
        for i in range(ns):
            for u0bit, kbit, k_hat_bit in zip(u0.subfunctions, ks[i].subfunctions,
                                              k_hat_s[i].subfunctions):
                u0bit += dtc * (BB[i] * kbit + B_hat[i] * k_hat_bit)

    # Last part of advance for the case where last implicit stage is new solution
    def _finalize_stiffly_accurate(self):
        khat, ghat, chat = self.kgchat
        u0 = self.u0
        for u0bit, ghatbit in zip(u0.subfunctions, ghat.subfunctions):
            u0bit.assign(ghatbit)

    def solver_stats(self):
        return self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations, self.num_mass_nonlinear_iterations, self.num_mass_linear_iterations
