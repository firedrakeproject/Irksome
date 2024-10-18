# formulate RK methods to solve for stage values rather than the stage derivatives.
from collections.abc import Iterable
from functools import reduce
from operator import mul

import numpy as np
from FIAT import Bernstein, ufc_simplex
from firedrake import (Function, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, TestFunction, dx,
                       inner, split)
from firedrake.petsc import PETSc
from numpy import vectorize
from ufl.classes import Zero
from ufl.constantvalue import as_ufl

from .bcs import stage2spaces4bc
from .ButcherTableaux import CollocationButcherTableau
from .manipulation import extract_terms, strip_dt_form
from .tools import (AI, IA, ConstantOrZero, MeshConstant, getNullspace, is_ode,
                    replace)


def isiterable(x):
    return hasattr(x, "__iter__") or isinstance(x, Iterable)


def split_field(num_fields, u):
    return np.array((u,) if num_fields == 1 else split(u), dtype="O")


def split_stage_field(num_stages, num_fields, UU):
    if num_fields == 1:
        if num_stages == 1:   # single-stage method
            UUbits = np.reshape(np.array((UU,), dtype='O'), (num_stages, num_fields))
        else:  # multi-stage methods
            UUbits = np.zeros((len(split(UU)),), dtype='O')
            for (i, x) in enumerate(split(UU)):
                UUbits[i] = np.zeros((1,), dtype='O')
                UUbits[i][0] = x
    else:
        UUbits = np.reshape(np.asarray(split(UU), dtype="O"), (num_stages, num_fields))
    return UUbits


def getBits(num_stages, num_fields, u0, UU, v, VV):
    u0bits, vbits = (split_field(num_fields, x) for x in (u0, v))
    UUbits, VVbits = (split_stage_field(num_stages, num_fields, x)
                      for x in (UU, VV))

    return u0bits, vbits, VVbits, UUbits


def getFormStage(F, butch, u0, t, dt, bcs=None, splitting=None, vandermonde=None,
                 bc_constraints=None, nullspace=None):
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
    :arg bc_constraints: optionally, a dictionary mapping (some of) the boundary
         conditions in `bcs` to triples of the form (params, lower, upper) indicating
         solver parameters to use and lower and upper bounds to provide in doing
         a bounds-constrained projection of the boundary data.
         Note: if these bounds change over time, the user is responsible for maintaining
         a handle on them and updating them between time steps.
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
    ZZ = Function(Vbig)

    # set up the pieces we need to work with to do our substitutions
    u0bits, vbits, VVbits, ZZbits = getBits(num_stages, num_fields,
                                            u0, ZZ, v, VV)

    MC = MeshConstant(V.mesh())
    vecconst = np.vectorize(lambda c: MC.Constant(c))

    C = vecconst(butch.c)
    A = vecconst(butch.A)

    veccorz = np.vectorize(lambda c: ConstantOrZero(c, MC))
    Vander = veccorz(vandermonde)
    Vander_inv = veccorz(np.linalg.inv(vandermonde))

    # convert from Bernstein to Lagrange representation
    # the Bernstein coefficients are [u0; ZZ], and the Lagrange
    # are [u0; UU] since the value at the left-endpoint is unchanged.
    # Since u0 is not part of the unknown vector of stages, we disassemble
    # the Vandermonde matrix (first row is [1, 0, ...])
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
    if bc_constraints is None:
        bc_constraints = {}
    bcsnew = []

    # For each BC, we need a new BC for each stage
    # so we need to figure out how the function is indexed (mixed + vec)
    # and then set it to have the value of the original argument at
    # time t+C[i]*dt.

    for bc in bcs:
        bcarg = as_ufl(bc._original_arg)

        if bc in bc_constraints:
            bcparams, bclower, bcupper = bc_constraints[bc]
            gcur = replace(bcarg, {t: t+C[i] * dt})
            gcur = gcur - Vander_col[i] * bcarg
        else:
            gdats_cur = np.zeros((num_stages,), dtype="O")
            for i in range(num_stages):
                Vbigi = stage2spaces4bc(bc, Vbig, i)
                gcur = replace(bcarg, {t: t+C[i]*dt})
                gcur = gcur - Vander_col[i] * bcarg
                gdats_cur[i] = gcur

            zdats_cur = Vander_inv[1:, 1:] @ gdats_cur

            bcnew_cur = []
            for i in range(num_stages):
                Vbigi = stage2spaces4bc(bc, Vbig, i)
                bcnew_cur.append(bc.reconstruct(V=Vbigi, g=zdats_cur[i]))

            bcsnew.extend(bcnew_cur)

    nspacenew = getNullspace(V, Vbig, butch, nullspace)

    # only form update stuff if we need it
    # which means neither stiffly accurate nor Vandermonde
    if not butch.is_stiffly_accurate:
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
        for bc in bcs:
            bcarg = as_ufl(bc._original_arg)
            gcur = replace(bcarg, {t: t+dt})
            update_bcs.append(bc.reconstruct(g=gcur))

        update_stuff = (unew, Fupdate, update_bcs)
    else:
        update_stuff = None

    return (Fnew, update_stuff, ZZ, bcsnew, nspacenew)


class StageValueTimeStepper:
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None, update_solver_parameters=None,
                 bc_constraints=None,
                 splitting=AI, basis_type=None,
                 nullspace=None, appctx=None):

        # we can only do DAE-type problems correctly if one assumes a stiffly-accurate method.
        assert is_ode(F, u0) or butcher_tableau.is_stiffly_accurate

        self.u0 = u0
        self.t = t
        self.dt = dt
        num_stages = self.num_stages = len(butcher_tableau.b)
        self.num_fields = len(u0.function_space())
        self.butcher_tableau = butcher_tableau
        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        if basis_type is None:
            vandermonde = None
        elif basis_type == "Bernstein":
            assert isinstance(butcher_tableau, CollocationButcherTableau), "Need collocation for Bernstein conversion"
            bern = Bernstein(ufc_simplex(1), num_stages)
            cc = np.reshape(np.append(0, butcher_tableau.c), (-1, 1))
            vandermonde = bern.tabulate(0, np.reshape(cc, (-1, 1)))[(0, )].T
        else:
            raise ValueError("Unknown or unimplemented basis transformation type")

        Fbig, update_stuff, UU, bigBCs, nsp = getFormStage(
            F, butcher_tableau, u0, t, dt, bcs, vandermonde=vandermonde,
            splitting=splitting)

        self.UU = UU
        self.bigBCs = bigBCs
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

        if (not butcher_tableau.is_stiffly_accurate) and (basis_type != "Bernstein"):
            unew, Fupdate, update_bcs = self.update_stuff
            self.update_problem = NonlinearVariationalProblem(
                Fupdate, unew, update_bcs)

            self.update_solver = NonlinearVariationalSolver(
                self.update_problem,
                solver_parameters=update_solver_parameters)
            self._update = self._update_general
        else:
            self._update = self._update_stiff_acc

        # stash these for later in case we do bounds constraints
        self.stage_lower_bound = Function(UU.function_space())
        self.stage_upper_bound = Function(UU.function_space())

    def _update_stiff_acc(self):
        u0 = self.u0
        u0bits = u0.subfunctions
        UUs = self.UU.subfunctions

        for i, u0bit in enumerate(u0bits):
            u0bit.assign(UUs[self.num_fields*(self.num_stages-1)+i])

    def _update_general(self):
        unew, Fupdate, update_bcs = self.update_stuff
        self.update_solver.solve()
        unewbits = unew.subfunctions
        for u0bit, unewbit in zip(self.u0.subfunctions, unewbits):
            u0bit.assign(unewbit)

    def advance(self, bounds=None):
        if bounds is None:
            stage_bounds = None
        else:
            bounds_type, lower, upper = bounds
            slb = self.stage_lower_bound
            sub = self.stage_upper_bound
            if bounds_type == "stage":
                if lower is None:
                    slb.assign(PETSc.NINFINITY)
                else:
                    for i in range(self.num_stages):
                        for j, lower_bit in enumerate(lower.subfunctions):
                            slb.subfunctions[i*self.num_fields+j].assign(lower_bit)
                if upper is None:
                    sub.assign(PETSc.INFINITY)
                else:
                    for i in range(self.num_stages):
                        for j, upper_bit in enumerate(upper.subfunctions):
                            sub.subfunctions[i*self.num_fields+j].assign(upper_bit)
            elif bounds_type == "last_stage":
                if lower is None:
                    slb.assign(PETSc.NINFINITY)
                else:
                    for i in range(self.num_stages-1):
                        for j in range(self.num_fields):
                            slb.subfunctions[i*self.num_fields+j].assign(PETSc.NINFINITY)
                    for j, lower_bit in enumerate(lower.subfunctions):
                        slb.subfunctions[-(self.num_fields-j)].assign(lower_bit)
                if upper is None:
                    sub.assign(PETSc.INFINITY)
                else:
                    for i in range(self.num_stages-1):
                        for j in range(self.num_fields):
                            sub.subfunctions[i*self.num_fields+j].assign(PETSc.INFINITY)
                    for j, upper_bit in enumerate(upper.subfunctions):
                        sub.subfunctions[-(self.num_fields-j)].assign(upper_bit)
            else:
                raise ValueError("Unknown bounds type")

            stage_bounds = (slb, sub)

        self.solver.solve(bounds=stage_bounds)

        self.num_steps += 1
        self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()

        self._update()

    def solver_stats(self):
        return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)
