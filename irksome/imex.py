import FIAT
import numpy as np
from firedrake import (DirichletBC, Function, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, TestFunction,
                       assemble, as_ufl, dx, inner, interpolate,
                       project, split)
from firedrake.dmhooks import pop_parent, push_parent
from ufl.classes import Zero

from .ButcherTableaux import RadauIIA
from .deriv import TimeDerivative
from .stage import getBits, getFormStage
from .tools import AI, IA, MeshConstant, replace


def riia_explicit_coeffs(k):
    """Computes the coefficients needed for the explicit part
    of a RadauIIA-IMEX method."""
    U = FIAT.ufc_simplex(1)
    L = FIAT.GaussRadau(U, k - 1)
    Q = FIAT.make_quadrature(L.ref_el, 2*k)
    c = np.asarray([list(ell.pt_dict.keys())[0][0]
                    for ell in L.dual.nodes])

    Q = FIAT.make_quadrature(L.ref_el, 2*k)
    qpts = Q.get_points()
    qwts = Q.get_weights()

    A = np.zeros((k, k))
    for i in range(k):
        qpts_i = 1 + qpts * c[i]
        qwts_i = qwts * c[i]
        Lvals_i = L.tabulate(0, qpts_i)[0, ]
        A[i, :] = Lvals_i @ qwts_i

    return A


def getFormExplicit(Fexp, butch, u0, UU, t, dt, splitting=None):
    """Processes the explicitly split-off part for a RadauIIA-IMEX
    method.  Returns the forms for both the iterator and propagator,
    which really just differ by which constants are in them."""
    v = Fexp.arguments()[0]
    V = v.function_space()
    msh = V.mesh()
    Vbig = UU.function_space()
    VV = TestFunction(Vbig)

    num_stages = butch.num_stages
    num_fields = len(V)
    MC = MeshConstant(msh)
    vc = np.vectorize(lambda c: MC.Constant(c))
    Aexp = riia_explicit_coeffs(num_stages)
    Aprop = vc(Aexp)
    Ait = vc(butch.A)
    C = vc(butch.c)

    u0bits, vbits, VVbits, UUbits = getBits(num_stages, num_fields,
                                            u0, UU, v, VV)

    Fit = Zero()
    Fprop = Zero()

    if splitting == AI:
        for i in range(num_stages):
            # replace test function
            repl = {}

            for k in range(num_fields):
                repl[vbits[k]] = VVbits[i][k]
                for ii in np.ndindex(vbits[k].ufl_shape):
                    repl[vbits[k][ii]] = VVbits[i][k][ii]

            Ftmp = replace(Fexp, repl)

            # replace the solution with stage values
            for j in range(num_stages):
                repl = {t: t + C[j] * dt}

                for k in range(num_fields):
                    repl[u0bits[k]] = UUbits[j][k]
                    for ii in np.ndindex(u0bits[k].ufl_shape):
                        repl[u0bits[k][ii]] = UUbits[j][k][ii]

                # and sum the contribution
                replF = replace(Ftmp, repl)
                Fit += Ait[i, j] * dt * replF
                Fprop += Aprop[i, j] * dt * replF
    elif splitting == IA:
        # diagonal contribution to iterator
        for i in range(num_stages):
            repl = {t: t+C[i]*dt}
            for j in range(num_fields):
                repl[u0bits[j]] = UUbits[i][j]
                repl[vbits[j]] = VVbits[i][j]

            # Also get replacements right for indexing.
            for j in range(num_fields):
                for ii in np.ndindex(u0bits[j].ufl_shape):
                    repl[u0bits[j][ii]] = UUbits[i][j][ii]
                    repl[vbits[j][ii]] = VVbits[i][j][ii]

            Fit += dt * replace(Fexp, repl)

        # dense contribution to propagator
        Ablah = vc(np.linalg.solve(butch.A, Aexp))

        for i in range(num_stages):
            # replace test function
            repl = {}

            for k in range(num_fields):
                repl[vbits[k]] = VVbits[i][k]
                for ii in np.ndindex(vbits[k].ufl_shape):
                    repl[vbits[k][ii]] = VVbits[i][k][ii]

            Ftmp = replace(Fexp, repl)

            # replace the solution with stage values
            for j in range(num_stages):
                repl = {t: t + C[j] * dt}

                for k in range(num_fields):
                    repl[u0bits[k]] = UUbits[j][k]
                    for ii in np.ndindex(u0bits[k].ufl_shape):
                        repl[u0bits[k][ii]] = UUbits[j][k][ii]

                # and sum the contribution
                Fprop += Ablah[i, j] * dt * replace(Ftmp, repl)
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
        Fbig, _, UU, bigBCs, nsp = getFormStage(
            F, butcher_tableau, u0, t, dt, bcs,
            splitting=splitting, nullspace=nullspace)

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


class BCThingy:
    def __init__(self):
        pass

    def __call__(self, u):
        return u


class BCCompOfNotMixedThingy:
    def __init__(self, comp):
        self.comp = comp

    def __call__(self, u):
        return u[self.comp]


class BCMixedBitThingy:
    def __init__(self, sub):
        self.sub = sub

    def __call__(self, u):
        return u.sub(self.sub)


class BCCompOfMixedBitThingy:
    def __init__(self, sub, comp):
        self.sub = sub
        self.comp = comp

    def __call__(self, u):
        return u.sub(self.sub)[self.comp]


def getThingy(V, bc):
    num_fields = len(V)
    Vbc = bc.function_space()
    if num_fields == 1:
        comp = Vbc.component
        if comp is None:
            return BCThingy()
        else:
            return BCCompOfNotMixedThingy(comp)
    else:
        sub = bc.function_space_index()
        comp = Vbc.component
        if comp is None:
            return BCMixedBitThingy(sub)
        else:
            return BCCompOfMixedBitThingy(sub, comp)


def getFormsIMEX(F, Fexp, butch, t, dt, u0, bcs=None):
    if bcs is None:
        bcs = []

    v = F.arguments()[0]
    V = v.function_space()
    msh = V.mesh()
    assert V == u0.function_space()

    num_fields = len(V)

    k = Function(V)
    g = Function(V)

    khat = Function(V)
    vhat = TestFunction(V)
    ghat = Function(V)

    if num_fields == 1:
        k_bits = [k]
        u0bits = [u0]
        gbits = [g]
        ghat_bits = [ghat]
    else:
        k_bits = np.array(split(k), dtype=object)
        u0bits = split(u0)
        gbits = split(g)
        ghat_bits = split(g)

    MC = MeshConstant(msh)
    c = MC.Constant(1.0)
    chat = MC.Constant(1.0)
    a = MC.Constant(1.0)

    repl = {t: t + c * dt}
    for u0bit, kbit, gbit in zip(u0bits, k_bits, gbits):
        repl[u0bit] = gbit + dt * a * kbit
        repl[TimeDerivative(u0bit)] = kbit
    stage_F = replace(F, repl)

    replhat = {t: t + chat * dt}
    for u0bit, ghatbit in zip(u0bits, ghat_bits):
        replhat[u0bit] = ghatbit
    Fhat = inner(khat, vhat)*dx - replace(Fexp, replhat)

    bcnew = []
    gblah = []

    for bc in bcs:
        Vbc = bc.function_space()
        bcarg = as_ufl(bc._original_arg)
        bcarg_stage = replace(bcarg, {t: t + c * dt})
        try:
            gdat = assemble(interpolate(bcarg, Vbc))
            gmethod = lambda gd, gc: gd.interpolate(gc)
        except:  # noqa: E722
            gdat = assemble(project(bcarg, Vbc))
            gmethod = lambda gd, gc: gd.project(gc)

        new_bc = DirichletBC(Vbc, gdat, bc.sub_domain)
        bcnew.append(new_bc)

        dat4bc = getThingy(V, bc)
        gdat2 = Function(gdat.function_space())
        gblah.append((gdat, gdat2, bcarg_stage, gmethod, dat4bc))

    return stage_F, (k, g, a, c), bcnew, gblah, Fhat, (khat, ghat, chat)


class DIRKIMEXMethod:
    """Front-end class for advancing a time-dependent PDE via a diagonally-implicit
    Runge-Kutta method formulated in terms of stage derivatives."""

    def __init__(self, F, F_explicit, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None, mass_parameters=None, appctx=None, nullspace=None):
        assert butcher_tableau.is_explicit_implicit

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        self.butcher_tableau = butcher_tableau
        self.V = V = u0.function_space()
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = butcher_tableau.num_stages
        self.ks = [Function(V) for _ in range(self.num_stages)]
        self.k_hat_s = [Function(V) for _ in range(self.num_stages+1)]

        stage_F, (k, g, a, c), bcnew, gblah, Fhat, (khat, ghat, chat) = getFormsIMEX(
            F, F_explicit, butcher_tableau, t, dt, u0, bcs=bcs)

        self.bcnew = bcnew
        self.gblah = gblah

        appctx_irksome = {"F": F,
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

    def advance(self):
        k, g, a, c = self.kgac
        khat, ghat, chat = self.kgchat
        ks = self.ks
        k_hat_s = self.k_hat_s
        u0 = self.u0
        dtc = float(self.dt)
        bt = self.butcher_tableau
        AA = bt.A
        A_hat = bt.A_hat
        CC = bt.c
        C_hat = bt.c_hat
        BB = bt.b
        B_hat = bt.b_hat

        # Calculate explicit term for the first stage
        ghat.assign(u0)
        chat.assign(C_hat[0])
        self.mass_solver.solve()
        k_hat_s[0].assign(khat)

        for i in range(self.num_stages):
            # Set implicit coefficients
            a.assign(AA[i, i])
            c.assign(CC[i])

            g.assign(u0)

            # Update g with contributions from previous stages
            for j in range(i):
                ksplit = ks[j].subfunctions
                for gbit, kbit in zip(g.subfunctions, ksplit):
                    gbit += dtc * AA[i, j] * kbit

            for j in range(i+1):
                k_hat_split = k_hat_s[j].subfunctions
                for gbit, k_hat_bit in zip(g.subfunctions, k_hat_split):
                    gbit += dtc * A_hat[i, j] * k_hat_bit

            # Solve for current stage
            for (bc, (gdat, gdat2, gcur, gmethod, dat4bc)) in zip(self.bcnew, self.gblah):
                gmethod(gdat, gcur)
                gmethod(gdat2, dat4bc(u0))
                gdat -= gdat2

                for j in range(i):
                    gmethod(gdat2, dat4bc(ks[j]))
                    gdat -= dtc * AA[i, j] * gdat2

                for j in range(i+1):
                    gmethod(gdat2, dat4bc(k_hat_s[j]))
                    gdat -= dtc * A_hat[i, j] * gdat2

                gdat /= dtc * AA[i, i]

            # Solve the nonlinear problem at stage i
            self.solver.solve()
            ks[i].assign(k)

            # Update the solution for next stage
            for ghatbit, gbit in zip(ghat.subfunctions, g.subfunctions):
                ghatbit.assign(gbit)
            for ghatbit, kbit in zip(ghat.subfunctions, ks[i].subfunctions):
                ghatbit += dtc * AA[i, i] * kbit
            chat.assign(C_hat[i+1])

            self.mass_solver.solve()
            k_hat_s[i + 1].assign(khat)

        # Final solution update
        for i in range(self.num_stages):
            for u0bit, kbit in zip(u0.subfunctions, ks[i].subfunctions):
                u0bit += dtc * BB[i] * kbit

        for i in range(self.num_stages+1):
            for u0bit, k_hat_bit in zip(u0.subfunctions, k_hat_s[i].subfunctions):
                u0bit += dtc * B_hat[i] * k_hat_bit

        self.num_steps += 1

    def solver_stats(self):
        return self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations
