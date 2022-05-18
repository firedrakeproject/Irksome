# formulate RK methods to solve for stage values rather than the stage derivatives.
# This gives a different flow for the substitutions and is a first step
# toward polynomial/imex-type methods
from functools import reduce
from operator import mul
import numpy as np

from firedrake import TestFunction, Function, split, Constant, DirichletBC, interpolate, project, NonlinearVariationalProblem, NonlinearVariationalSolver, inner, dx

from .tools import replace, getNullspace
from .manipulation import extract_terms, strip_dt_form
from ufl.classes import Zero
from .ButcherTableaux import RadauIIA
from numpy import vectorize


def getFormStage(F, butch, u0, t, dt, bcs=None, splitting=None,
                 nullspace=None):
    v = F.arguments()[0]
    V = v.function_space()

    assert V == u0.function_space()

    num_stages = butch.num_stages
    num_fields = len(V)

    # s-way product space for the stage variables
    Vbig = reduce(mul, (V for _ in range(num_stages)))

    VV = TestFunction(Vbig)
    UU = Function(Vbig)

    vecconst = np.vectorize(Constant)
    C = vecconst(butch.c)
    A = vecconst(butch.A)

    # set up the pieces we need to work with to do our substitutions

    nsxnf = (num_stages, num_fields)

    if num_fields == 1:
        u0bits = np.array([u0])
        vbits = np.array([v])
        if num_stages == 1:   # single-stage method
            VVbits = np.array([[VV]])
            UUbits = np.array([[UU]])
        else:  # multi-stage methods
            VVbits = np.reshape(split(VV), nsxnf)
            UUbits = np.reshape(split(UU), nsxnf)
    else:
        u0bits = np.array(list(split(u0)))
        vbits = np.array(list(split(v)))
        VVbits = np.reshape(split(VV), nsxnf)
        UUbits = np.reshape(split(UU), nsxnf)

    split_form = extract_terms(F)

    Fnew = Zero()

    # print(split_form.time)
    # print(split_form.remainder)

    # first, process terms with a time derivative.  I'm
    # assuming we have something of the form inner(Dt(g(u0)), v)*dx
    # For each stage i, this gets replaced with
    # inner((g(stages[i]) - g(u0))/dt, v)*dx
    # but we have to carefully handle the cases where g indexes into
    # pieces of u
    dtless = strip_dt_form(split_form.time)

    for i in range(num_stages):
        repl = {t: t+C[i]*dt}
        for j in range(num_fields):
            repl[u0bits[j]] = (UUbits[i][j] - u0bits[j]) / dt
            repl[vbits[j]] = VVbits[i][j]

        # Also get replacements right for indexing.
        for j in range(num_fields):
            for ii in np.ndindex(u0bits[j].ufl_shape):
                repl[u0bits[j][ii]] = UUbits[i][j][ii] - u0bits[j][ii]
                repl[vbits[j][ii]] = VVbits[i][j][ii]

        Fnew += replace(dtless, repl)

    # Now for the non-time derivative parts
    for i in range(num_stages):
        # replace test function
        repl = {}

        for k in range(num_fields):
            repl[vbits[k]] = VVbits[i][k]
            for ii in np.ndindex(vbits[k].ufl_shape):
                repl[vbits[k][ii]] = VVbits[i][k][ii]

        Ftmp = replace(split_form.remainder, repl)

        # replace the solution with stage values
        for j in range(num_stages):
            repl = {t: t + C[j] * dt}

            for k in range(num_fields):
                repl[u0bits[k]] = UUbits[j][k]
                for ii in np.ndindex(u0bits[k].ufl_shape):
                    repl[u0bits[k][ii]] = UUbits[j][k][ii]

            # and sum the contribution
            Fnew += A[i, j] * dt * replace(Ftmp, repl)

    if bcs is None:
        bcs = []
    bcsnew = []
    gblah = []

    # For each BC, we need a new BC for each stage
    # so we need to figure out how the function is indexed (mixed + vec)
    # and then set it to have the value of the original argument at
    # time t+C[i]*dt.

    for bc in bcs:
        if num_fields == 1:  # not mixed space
            comp = bc.function_space().component
            if comp is not None:  # check for sub-piece of vector-valued
                Vsp = V.sub(comp)
                Vbigi = lambda i: Vbig[i].sub(comp)
            else:
                Vsp = V
                Vbigi = lambda i: Vbig[i]
        else:  # mixed space
            sub = bc.function_space_index()
            comp = bc.function_space().component
            if comp is not None:  # check for sub-piece of vector-valued
                Vsp = V.sub(sub).sub(comp)
                Vbigi = lambda i: Vbig[sub+num_fields*i].sub(comp)
            else:
                Vsp = V.sub(sub)
                Vbigi = lambda i: Vbig[sub+num_fields*i]

        bcarg = bc._original_arg
        for i in range(num_stages):
            try:
                gdat = interpolate(bcarg, Vsp)
                gmethod = lambda gd, gc: gd.interpolate(gc)
            except:  # noqa: E722
                gdat = project(bcarg, Vsp)
                gmethod = lambda gd, gc: gd.project(gc)

            gcur = replace(bcarg, {t: t+C[i]*dt})
            bcsnew.append(DirichletBC(Vbigi(i), gdat, bc.sub_domain))
            gblah.append((gdat, gcur, gmethod))

    nspacenew = getNullspace(V, Vbig, butch, nullspace)

    # For RIIA, we have an optimized update rule and don't need to
    # build the variational form for doing updates.
    if isinstance(butch, RadauIIA):
        return Fnew, None, UU, bcsnew, gblah, nspacenew

    # Otherwise...
    unew = Function(V)

    Fupdate = inner(unew - u0, v) * dx
    B = vectorize(Constant)(butch.b)
    C = vectorize(Constant)(butch.c)

    for i in range(num_stages):
        repl = {t: t + C[i] * dt}

        for k in range(num_fields):
            repl[u0bits[k]] = UUbits[i][k]
            for ii in np.ndindex(u0bits[k].ufl_shape):
                repl[u0bits[k][ii]] = UUbits[i][k][ii]

        eFFi = replace(split_form.remainder, repl)

        Fupdate += dt * B[i] * eFFi

    # And the BC's for the update -- just the original BC at t+dt
    update_bcs = []
    update_bcs_gblah = []
    for bc in bcs:
        if num_fields == 1:  # not mixed space
            comp = bc.function_space().component
            if comp is not None:  # check for sub-piece of vector-valued
                Vsp = V.sub(comp)
            else:
                Vsp = V
        else:  # mixed space
            sub = bc.function_space_index()
            comp = bc.function_space().component
            if comp is not None:  # check for sub-piece of vector-valued
                Vsp = V.sub(sub).sub(comp)
            else:
                Vsp = V.sub(sub)

        bcarg = bc._original_arg
        try:
            gdat = interpolate(bcarg, Vsp)
            gmethod = lambda gd, gc: gd.interpolate(gc)
        except:  # noqa: E722
            gdat = project(bcarg, Vsp)
            gmethod = lambda gd, gc: gd.project(gc)

        gcur = replace(bcarg, {t: t+dt})
        update_bcs.append(DirichletBC(Vsp, gdat, bc.sub_domain))
        update_bcs_gblah.append((gdat, gcur, gmethod))

        return (Fnew, (unew, Fupdate, update_bcs, update_bcs_gblah),
                UU, bcsnew, gblah, nspacenew)


class StageValueTimeStepper:
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None, update_solver_parameters=None,
                 nullspace=None, appctx=None):
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = len(butcher_tableau.b)
        self.butcher_tableau = butcher_tableau

        Fbig, update_stuff, UU, bigBCs, gblah, nsp = getFormStage(
            F, butcher_tableau, u0, t, dt, bcs)

        self.UU = UU
        self.bigBCs = bigBCs
        self.bcdat = gblah
        self.update_stuff = update_stuff

        self.prob = NonlinearVariationalProblem(Fbig, UU, bigBCs)

        appctx_irksome = {"F": F,
                          "butcher_tableau": butcher_tableau,
                          "t": t,
                          "dt": dt,
                          "u0": u0,
                          "bcs": bcs,
                          "nullspace": nullspace}
        if appctx is None:
            appctx = appctx_irksome
        else:
            appctx = {**appctx, **appctx_irksome}

        self.solver = NonlinearVariationalSolver(
            self.prob, appctx=appctx, nullspace=nsp,
            solver_parameters=solver_parameters)

        if isinstance(butcher_tableau, RadauIIA):
            self._update = self._update_riia
        else:
            unew, Fupdate, update_bcs, update_bcs_gblah = self.update_stuff
            self.update_problem = NonlinearVariationalProblem(
                Fupdate, unew, update_bcs)

            self.update_solver = NonlinearVariationalSolver(
                self.update_problem,
                solver_parameters=update_solver_parameters)

            self._update = self._update_general

    def _update_riia(self):
        u0 = self.u0

        UUs = self.UU.split()
        nf = self.num_fields
        ns = self.num_stages

        for i, u0d in enumerate(u0.dat):
            u0d.data[:] = UUs[nf*(ns-1)+i].dat.data_ro[:]

    def _update_general(self):
        (unew, Fupdate, update_bcs, update_bcs_gblah) = self.update_stuff
        for gdat, gcur, gmethod in update_bcs_gblah:
            gmethod(gdat, gcur)
        self.update_solver.solve()
        for u0d, und in zip(self.u0.dat, unew.dat):
            u0d.data[:] = und.data_ro[:]

    def advance(self):
        for gdat, gcur, gmethod in self.bcdat:
            gmethod(gdat, gcur)

        self.solver.solve()

        self._update()