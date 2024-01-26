# formulate RK methods to solve for stage values rather than the stage derivatives.
from functools import reduce
from operator import mul

import numpy as np
from firedrake import (Constant, DirichletBC, Function,
                       NonlinearVariationalProblem, NonlinearVariationalSolver,
                       TestFunction, assemble, dx, inner, project, split)
from firedrake.__future__ import interpolate
from numpy import vectorize
from ufl.classes import Zero
from ufl.constantvalue import as_ufl

from .manipulation import extract_terms, strip_dt_form
from .tools import AI, IA, getNullspace, is_ode, replace


def getBits(num_stages, num_fields, u0, UU, v, VV):
    nsxnf = (num_stages, num_fields)
    if num_fields == 1:
        u0bits = np.zeros((1,), dtype='O')
        u0bits[0] = u0
        vbits = np.zeros((1,), dtype='O')
        vbits[0] = v
        if num_stages == 1:   # single-stage method
            VVbits = np.zeros((1,), dtype='O')
            VVbits[0] = np.zeros((1,), dtype='O')
            VVbits[0][0] = VV
            UUbits = np.zeros((1,), dtype='O')
            UUbits[0] = np.zeros((1,), dtype='O')
            UUbits[0][0] = UU
        else:  # multi-stage methods
            VVbits = np.zeros((len(split(VV)),), dtype='O')
            for (i, x) in enumerate(split(VV)):
                VVbits[i] = np.zeros((1,), dtype='O')
                VVbits[i][0] = x
            UUbits = np.zeros((len(split(UU)),), dtype='O')
            for (i, x) in enumerate(split(UU)):
                UUbits[i] = np.zeros((1,), dtype='O')
                UUbits[i][0] = x
    else:
        u0bits = np.array(list(split(u0)), dtype="O")
        vbits = np.array(list(split(v)), dtype="O")
        VVbits = np.reshape(np.asarray(split(VV), dtype="O"), nsxnf)
        UUbits = np.reshape(np.asarray(split(UU), dtype="O"), nsxnf)

    return u0bits, vbits, VVbits, UUbits


def getFormStage(F, butch, u0, t, dt, bcs=None, splitting=None,
                 nullspace=None):
    """Given a time-dependent variational form and a
    :class:`ButcherTableau`, produce UFL for the s-stage RK method.

    :arg F: UFL form for the semidiscrete ODE/DAE
    :arg butch: the :class:`ButcherTableau` for the RK method being used to
         advance in time.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg t: a :class:`Constant` referring to the current time level.
         Any explicit time-dependence in F is included
    :arg dt: a :class:`Constant` referring to the size of the current
         time step.
    :arg splitting: a callable that maps the (floating point) Butcher matrix
         a to a pair of matrices `A1, A2` such that `butch.A = A1 A2`.  This is used
         to vary between the classical RK formulation and Butcher's reformulation
         that leads to a denser mass matrix with block-diagonal stiffness.
         Only `AI` and `IA` are currently supported.
    :arg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
         containing (possible time-dependent) boundary conditions imposed
         on the system.
    :arg nullspace: A list of tuples of the form (index, VSB) where
         index is an index into the function space associated with `u`
         and VSB is a :class: `firedrake.VectorSpaceBasis` instance to
         be passed to a `firedrake.MixedVectorSpaceBasis` over the
         larger space associated with the Runge-Kutta method

    On output, we return a tuple consisting of several parts:

       - Fnew, the :class:`Form`
       - possibly a 4-tuple containing information needed to solve a mass matrix to update
         the solution (this is empty for RadauIIA methods for which there is a trivial
         update function.
       - UU, the :class:`firedrake.Function` holding all the stage time values.
         It lives in a :class:`firedrake.FunctionSpace` corresponding to the
         s-way tensor product of the space on which the semidiscrete
         form lives.
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the stages,
       - 'nspnew', the :class:`firedrake.MixedVectorSpaceBasis` object
         that represents the nullspace of the coupled system
       - `gblah`, a list of tuples of the form (f, expr, method),
         where f is a :class:`firedrake.Function` and expr is a
         :class:`ufl.Expr`.  At each time step, each expr needs to be
         re-interpolated/projected onto the corresponding f in order
         for Firedrake to pick up that time-dependent boundary
         conditions need to be re-applied.  The
         interpolation/projection is encoded in method, which is
         either `f.interpolate(expr-c*u0)` or `f.project(expr-c*u0)`, depending
         on whether the function space for f supports interpolation or
         not.
    """
    v = F.arguments()[0]
    V = v.function_space()

    assert V == u0.function_space()

    num_stages = butch.num_stages
    num_fields = len(V)

    # s-way product space for the stage variables
    Vbig = reduce(mul, (V for _ in range(num_stages)))
    VV = TestFunction(Vbig)
    UU = Function(Vbig)

    # set up the pieces we need to work with to do our substitutions
    u0bits, vbits, VVbits, UUbits = getBits(num_stages, num_fields,
                                            u0, UU, v, VV)

    vecconst = np.vectorize(lambda c: Constant(c, domain=V.mesh()))
    C = vecconst(butch.c)
    A = vecconst(butch.A)

    split_form = extract_terms(F)

    Fnew = Zero()

    # first, process terms with a time derivative.  I'm
    # assuming we have something of the form inner(Dt(g(u0)), v)*dx
    # For each stage i, this gets replaced with
    # inner((g(stages[i]) - g(u0))/dt, v)*dx
    # but we have to carefully handle the cases where g indexes into
    # pieces of u
    dtless = strip_dt_form(split_form.time)

    if splitting is None or splitting == AI:
        for i in range(num_stages):
            repl = {t: t+C[i]*dt}
            for j in range(num_fields):
                repl[u0bits[j]] = UUbits[i][j] - u0bits[j]
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

    elif splitting == IA:
        Ainv = np.vectorize(lambda c: Constant(c, domain=V.mesh()))(np.linalg.inv(butch.A))

        # time derivative part gets inverse of Butcher matrix.
        for i in range(num_stages):
            repl = {}

            for k in range(num_fields):
                repl[vbits[k]] = VVbits[i][k]
                for ii in np.ndindex(vbits[k].ufl_shape):
                    repl[vbits[k][ii]] = VVbits[i][k][ii]

            Ftmp = replace(dtless, repl)

            for j in range(num_stages):
                repl = {t: t + C[j] * dt}

                for k in range(num_fields):
                    repl[u0bits[k]] = (UUbits[j][k]-u0bits[k])
                    for ii in np.ndindex(u0bits[k].ufl_shape):
                        repl[u0bits[k][ii]] = UUbits[j][k][ii] - u0bits[k][ii]
                Fnew += Ainv[i, j] * replace(Ftmp, repl)
        # rest of the operator: just diagonal!
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

            Fnew += dt * replace(split_form.remainder, repl)
    else:
        raise NotImplementedError("Can't handle that splitting type")

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

        bcarg = as_ufl(bc._original_arg)
        for i in range(num_stages):
            try:
                gdat = assemble(interpolate(bcarg, Vsp))
                gmethod = lambda gd, gc: gd.interpolate(gc)
            except:  # noqa: E722
                gdat = project(bcarg, Vsp)
                gmethod = lambda gd, gc: gd.project(gc)

            gcur = replace(bcarg, {t: t+C[i]*dt})
            bcsnew.append(DirichletBC(Vbigi(i), gdat, bc.sub_domain))
            gblah.append((gdat, gcur, gmethod))

    nspacenew = getNullspace(V, Vbig, butch, nullspace)

    unew = Function(V)

    Fupdate = inner(unew - u0, v) * dx
    B = vectorize(lambda c: Constant(c, domain=V.mesh()))(butch.b)
    C = vectorize(lambda c: Constant(c, domain=V.mesh()))(butch.c)

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

        bcarg = as_ufl(bc._original_arg)
        try:
            gdat = assemble(interpolate(bcarg, Vsp))
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
                 splitting=AI,
                 nullspace=None, appctx=None):
        # we can only do DAE-type problems correctly if one assumes a stiffly-accurate method.
        ode_huh = is_ode(F, u0)
        stiff_acc_huh = butcher_tableau.is_stiffly_accurate
        assert ode_huh or stiff_acc_huh

        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = len(butcher_tableau.b)
        self.butcher_tableau = butcher_tableau
        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        Fbig, update_stuff, UU, bigBCs, gblah, nsp = getFormStage(
            F, butcher_tableau, u0, t, dt, bcs,
            splitting=splitting)

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
                          "splitting": splitting,
                          "nullspace": nullspace}
        if appctx is None:
            appctx = appctx_irksome
        else:
            appctx = {**appctx, **appctx_irksome}

        self.solver = NonlinearVariationalSolver(
            self.prob, appctx=appctx, nullspace=nsp,
            solver_parameters=solver_parameters)

        unew, Fupdate, update_bcs, update_bcs_gblah = self.update_stuff
        self.update_problem = NonlinearVariationalProblem(
            Fupdate, unew, update_bcs)

        self.update_solver = NonlinearVariationalSolver(
            self.update_problem,
            solver_parameters=update_solver_parameters)

        if stiff_acc_huh:
            self._update = self._update_stiff_acc
        else:
            self._update = self._update_general

    def _update_stiff_acc(self):
        u0 = self.u0

        UUs = self.UU.subfunctions
        nf = self.num_fields
        ns = self.num_stages

        u0bits = u0.subfunctions
        for i, u0bit in enumerate(u0bits):
            u0bit.assign(UUs[nf*(ns-1)+i])

    def _update_general(self):
        (unew, Fupdate, update_bcs, update_bcs_gblah) = self.update_stuff
        for gdat, gcur, gmethod in update_bcs_gblah:
            gmethod(gdat, gcur)
        self.update_solver.solve()
        u0bits = self.u0.subfunctions
        unewbits = unew.subfunctions
        for u0bit, unewbit in zip(u0bits, unewbits):
            u0bit.assign(unewbit)

    def advance(self):
        for gdat, gcur, gmethod in self.bcdat:
            gmethod(gdat, gcur)

        self.solver.solve()

        self.num_steps += 1
        self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()

        self._update()

    def solver_stats(self):
        return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)
