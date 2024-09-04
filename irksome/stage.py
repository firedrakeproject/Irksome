# formulate RK methods to solve for stage values rather than the stage derivatives.
from collections.abc import Iterable
from functools import reduce
from operator import mul

import numpy as np
from FIAT import Bernstein, ufc_simplex
from firedrake import (DirichletBC, Function, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, TestFunction, assemble, dx,
                       inner, project, split)
from firedrake.__future__ import interpolate
from firedrake.petsc import PETSc
from numpy import vectorize
from ufl.classes import Zero
from ufl.constantvalue import as_ufl

from .manipulation import extract_terms, strip_dt_form
from .tools import (AI, IA, ConstantOrZero, MeshConstant, getNullspace, is_ode,
                    replace, stage2spaces4bc)


def isiterable(x):
    return hasattr(x, "__iter__") or isinstance(x, Iterable)


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


def getFormStage(F, butch, u0, t, dt, bcs=None, splitting=None, vandermonde=None,
                 nullspace=None):
    """Given a time-dependent variational form and a
    :class:`ButcherTableau`, produce UFL for the s-stage RK method.

    :arg F: UFL form for the semidiscrete ODE/DAE
    :arg butch: the :class:`ButcherTableau` for the RK method being used to
         advance in time.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg splitting: a callable that maps the (floating point) Butcher matrix
         a to a pair of matrices `A1, A2` such that `butch.A = A1 A2`.  This is used
         to vary between the classical RK formulation and Butcher's reformulation
         that leads to a denser mass matrix with block-diagonal stiffness.
         Only `AI` and `IA` are currently supported.
    :arg vandermonde: a numpy array encoding a change of basis to the Lagrange
         polynomials associated with the collocation nodes from some other
         (e.g. Bernstein or Chebyshev) basis.  This allows us to solve for the
         coefficients in some basis rather than the values at particular stages,
         which can be useful for satisfying bounds constraints.
         If none is provided, we assume it is the identity, working in the
         Lagrange basis.
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

    # default to no basis transformation, identity matrix
    if vandermonde is None:
        vandermonde = np.eye(num_stages + 1)

    # s-way product space for the stage variables
    Vbig = reduce(mul, (V for _ in range(num_stages)))
    VV = TestFunction(Vbig)
    # UU = Function(Vbig)
    ZZ = Function(Vbig)

    # set up the pieces we need to work with to do our substitutions
    # u0bits, vbits, VVbits, UUbits = getBits(num_stages, num_fields,
    #                                         u0, UU, v, VV)

    u0bits, vbits, VVbits, ZZbits = getBits(num_stages, num_fields,
                                            u0, ZZ, v, VV)

    MC = MeshConstant(V.mesh())
    vecconst = np.vectorize(lambda c: MC.Constant(c))

    C = vecconst(butch.c)
    A = vecconst(butch.A)

    veccorz = np.vectorize(lambda c: ConstantOrZero(c, MC))
    Vander = veccorz(vandermonde)
    Vander_inv = veccorz(np.linalg.inv(vandermonde))

    # Note that the structure of the Vandermonde matrix means the first entry is unchanged.
    # it's the value at the previous time step, and we really want the action on the stage variables.
    # So to multiply [u0bits; UUbits] = Vander @ [u0bits; UUbits],
    # we partition Vander accordingly to do this computation.x

    Vander_col = Vander[1:, 0]
    Vander0 = Vander[1:, 1:]

    v0u0 = np.zeros((num_stages, num_fields), dtype="O")
    for i in range(num_stages):
        for j in range(num_fields):
            v0u0[i, j] = Vander_col[i] * u0bits[j]

    if num_fields == 1:
        v0u0 = np.reshape(v0u0, (-1,))

    UUbits = v0u0 + Vander0 @ ZZbits

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
        Ainv = np.vectorize(lambda c: MC.Constant(c))(np.linalg.inv(butch.A))

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
        bcarg = as_ufl(bc._original_arg)

        gblah_cur = []

        for i in range(num_stages):
            Vsp, Vbigi = stage2spaces4bc(bc, V, Vbig, i)
            try:
                gdat = assemble(interpolate(bcarg, Vsp))
                gmethod = lambda gd, gc: gd.interpolate(gc)
            except:  # noqa: E722
                gdat = project(bcarg, Vsp)
                gmethod = lambda gd, gc: gd.project(gc)

            gcur = replace(bcarg, {t: t+C[i]*dt})
            gblah_cur.append((gdat, gcur - Vander_col[i] * bcarg, gmethod))

        gdats_cur = np.zeros((num_stages,), dtype="O")
        for i in range(num_stages):
            gdats_cur[i] = gblah_cur[i][0]

        zdats_cur = Vander_inv[1:, 1:] @ gdats_cur

        bcnew_cur = []
        for i in range(num_stages):
            Vsp, Vbigi = stage2spaces4bc(bc, V, Vbig, i)
            bcnew_cur.append(DirichletBC(Vbigi, zdats_cur[i], bc.sub_domain))

        bcsnew.extend(bcnew_cur)
        gblah.extend(gblah_cur)

    nspacenew = getNullspace(V, Vbig, butch, nullspace)

    unew = Function(V)

    Fupdate = inner(unew - u0, v) * dx
    B = vectorize(lambda c: MC.Constant(c))(butch.b)
    C = vectorize(lambda c: MC.Constant(c))(butch.c)

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
            ZZ, bcsnew, gblah, nspacenew)


class StageValueTimeStepper:
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None, update_solver_parameters=None,
                 splitting=AI, basis_type=None,
                 nullspace=None, appctx=None,
                 bounds=None):

        # we can only do DAE-type problems correctly if one assumes a stiffly-accurate method.
        ode_huh = is_ode(F, u0)
        stiff_acc_huh = butcher_tableau.is_stiffly_accurate
        assert ode_huh or stiff_acc_huh

        # validate bounds
        if bounds is not None:
            assert bounds[0] in ["stage", "last_stage", "time_level"]

        self.u0 = u0
        self.t = t
        self.dt = dt
        num_fields = self.num_fields = len(u0.function_space())
        num_stages = self.num_stages = len(butcher_tableau.b)
        self.butcher_tableau = butcher_tableau
        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        if basis_type is None:
            vandermonde = None
        elif basis_type == "Bernstein":
            # assert self.num_stages > 1, ValueError("Bernstein only defined for degree >= 1")
            bern = Bernstein(ufc_simplex(1), self.num_stages)
            cc = np.reshape(np.append(0, butcher_tableau.c), (-1, 1))
            vandermonde = bern.tabulate(0, np.reshape(cc, (-1, 1)))[0,].T
        else:
            raise ValueError("Unknown or unimplemented basis transformation type")

        Fbig, update_stuff, UU, bigBCs, gblah, nsp = getFormStage(
            F, butcher_tableau, u0, t, dt, bcs, vandermonde=vandermonde,
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

        if bounds is not None and bounds[0] == "time_level":
            assert update_solver_parameters is not None, \
                "update_solver_parameters required for bounds-constrained update"

        self.update_solver = NonlinearVariationalSolver(
            self.update_problem,
            solver_parameters=update_solver_parameters)

        if bounds is not None:
            V = u0.function_space()
            bounds_type, lower, upper = bounds
            lower_func = Function(V)
            upper_func = Function(V)
            if not isiterable(lower):
                lower = (lower,)
            if not isiterable(upper):
                upper = (upper,)
            for l, lf in zip(lower, lower_func.subfunctions):
                if l is not None:
                    lf.assign(l)
                else:
                    lf.assign(PETSc.NINFINITY)
            for u, uf in zip(upper, upper_func.subfunctions):
                if u is not None:
                    uf.assign(u)
                else:
                    uf.assign(PETSc.INFINITY)

            if bounds_type in ["stage", "last_stage"]:
                Vbig = UU.function_space()
                stage_lower = Function(Vbig)
                stage_upper = Function(Vbig)

            if bounds_type == "stage":
                for i in range(num_stages):
                    for j in range(num_fields):
                        stage_lower.subfunctions[i*num_fields+j].assign(lower_func.subfunctions[j])
                        stage_upper.subfunctions[i*num_fields+j].assign(upper_func.subfunctions[j])
            elif bounds_type == "last_stage":
                for i in range(num_stages-1):
                    for j in range(num_fields):
                        stage_lower.subfunctions[i*num_fields+j].assign(PETSc.NINFINITY)
                        stage_upper.subfunctions[i*num_fields+j].assign(PETSc.INFINITY)
                offset = (num_stages-1) * num_fields
                for j in range(num_fields):
                    stage_lower.subfunctions[offset+j].assign(lower_func.subfunctions[j])
                    stage_upper.subfunctions[offset+j].assign(upper_func.subfunctions[j])

            if bounds_type in ["stage", "last_stage"]:
                self._stage_solve = lambda: self.solver.solve(bounds=(stage_lower, stage_upper))
                self._update = self._update_stiff_acc if stiff_acc_huh else self._update_general
                self._update_solve = self.update_solver.solve

            elif bounds_type == "time_level":
                if not ode_huh:
                    raise NotImplementedError(
                        """Time-level bounds constraints not working for DAE-type problems.  Please try with "bounds_type" set to "last_stage".""")
                self._stage_solve = self.solver.solve
                self._update_solve = lambda: self.update_solver.solve(bounds=(lower_func, upper_func))
                self._update = self._update_general
        else:  # no bounds constraint
            self._stage_solve = self.solver.solve
            self._update_solve = self.update_solver.solve
            self._update = self._update_stiff_acc if stiff_acc_huh else self._update_general

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
        self._update_solve()
        u0bits = self.u0.subfunctions
        unewbits = unew.subfunctions
        for u0bit, unewbit in zip(u0bits, unewbits):
            u0bit.assign(unewbit)

    def advance(self):
        for gdat, gcur, gmethod in self.bcdat:
            gmethod(gdat, gcur)

        self._stage_solve()

        self.num_steps += 1
        self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()

        self._update()

    def solver_stats(self):
        return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)
