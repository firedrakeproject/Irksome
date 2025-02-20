import numpy
from firedrake import Function
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake import split
from ufl.constantvalue import as_ufl
from ufl.algorithms.analysis import extract_type

from .deriv import TimeDerivative
from .tools import replace, MeshConstant
from .bcs import bc2space


def getFormDIRK(F, ks, butch, t, dt, u0, bcs=None):
    if bcs is None:
        bcs = []

    v = F.arguments()[0]
    V = v.function_space()
    msh = V.mesh()
    assert V == u0.function_space()

    num_fields = len(V)
    num_stages = butch.num_stages
    k = Function(V)
    g = Function(V)

    # If we're on a mixed problem, we need to replace pieces of the
    # solution.  Stores an array of the splittings of the k for each stage.
    if num_fields == 1:
        k_bits = [k]
        u0bits = [u0]
        gbits = [g]
    else:
        k_bits = numpy.array(split(k), dtype=object)
        u0bits = split(u0)
        gbits = split(g)

    # Tracks which subfunctions have derivatives and need to be evolved
    has_deriv = numpy.array([], dtype=int)
    no_deriv = numpy.array([], dtype=int)

    # Note: the Constant c is used for substitution in both the
    # variational form and BC's, and we update it for each stage in
    # the loop over stages in the advance method.  The Constant a is
    # used similarly in the variational form
    MC = MeshConstant(msh)
    c = MC.Constant(1.0)
    a = MC.Constant(1.0)

    deriv_list = extract_type(F, TimeDerivative)

    repl = {t: t+c*dt}
    for i, (u0bit, kbit, gbit) in enumerate(zip(u0bits, k_bits, gbits)):
        if TimeDerivative(u0bit) in deriv_list:
            repl[u0bit] = gbit + dt * a * kbit
            repl[TimeDerivative(u0bit)] = kbit
            has_deriv = numpy.append(has_deriv, i)
        else:
            repl[u0bit] = kbit
            no_deriv = numpy.append(no_deriv, i)

    if has_deriv.size == 0:
        raise TypeError("Form must have at least one TimeDerivative")

    stage_F = replace(F, repl)

    bcnew = []

    # For the DIRK case, we need one new BC for each old one (rather
    # than one per stage), but we need a `Function` inside of each BC
    # and a rule for computing that function at each time for each
    # stage.
    a_vals = numpy.array([MC.Constant(0) for i in range(num_stages)],
                         dtype=object)
    d_val = MC.Constant(1.0)
    for bc in bcs:
        bcarg = as_ufl(bc._original_arg)
        bcarg_stage = replace(bcarg, {t: t+c*dt})
        if bcarg_stage == 0:
            # Homogeneous BC, just zero out stage dofs
            bcnew.append(bc.reconstruct(g=0))
            continue

        gdat = bcarg_stage - bc2space(bc, u0)
        for i in range(num_stages):
            gdat -= dt*a_vals[i]*bc2space(bc, ks[i])

        gdat /= dt*d_val

        bcnew.append(bc.reconstruct(g=gdat))

    return stage_F, (k, g, a, c), bcnew, (a_vals, d_val), (has_deriv, no_deriv)


class DIRKTimeStepper:
    """Front-end class for advancing a time-dependent PDE via a diagonally-implicit
    Runge-Kutta method formulated in terms of stage derivatives."""

    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None,
                 appctx=None, nullspace=None):
        assert butcher_tableau.is_diagonally_implicit

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        self.butcher_tableau = butcher_tableau
        self.num_stages = num_stages = butcher_tableau.num_stages
        self.AAb = numpy.vstack((butcher_tableau.A, butcher_tableau.b))
        self.CCone = numpy.append(butcher_tableau.c, 1.0)

        # Need to be able to set BCs for either the DIRK or explicit cases.

        # For DIRK, we say that the stage i solution should match the
        # prescribed boundary values at time c[i], which means we use
        # the i^th row of the Butcher tableau in determining the
        # boundary condition

        # For explicit, we say that the stage i solution should be
        # determined to match the prescribed boundary values at time
        # c[i+1] (the first stage where it appears), which means we
        # use the (i+1)^st row of the Butcher tableau in determining
        # the boundary condition, and the full reconstruction for the
        # final stage

        if butcher_tableau.is_explicit:
            self.AAb = self.AAb[1:]
            self.CCone = self.CCone[1:]

        self.V = V = u0.function_space()
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.ks = [Function(V) for _ in range(num_stages)]

        # "k" is a generic function for which we will solve the
        # NVLP for the next stage value
        # "ks" is a list of functions for the stage values
        # that we update as we go.  We need to remember the
        # stage values we've computed earlier in the time step...

        stage_F, (k, g, a, c), bcnew, (a_vals, d_val), (has_deriv, no_deriv) \
            = getFormDIRK(F, self.ks, butcher_tableau, t, dt, u0, bcs=bcs)

        self.bcnew = bcnew

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

        self.problem = NLVP(stage_F, k, bcnew)
        self.solver = NLVS(
            self.problem, appctx=appctx, solver_parameters=solver_parameters,
            nullspace=nullspace)

        self.kgac = k, g, a, c
        self.bc_constants = a_vals, d_val
        self.derivs = (has_deriv, no_deriv)

    def update_bc_constants(self, i, c):
        AAb = self.AAb
        CCone = self.CCone
        a_vals, d_val = self.bc_constants
        ns = AAb.shape[1]
        for j in range(i):
            a_vals[j].assign(AAb[i, j])
        for j in range(i, ns):
            a_vals[j].assign(0)
        d_val.assign(AAb[i, i])
        c.assign(CCone[i])

    def advance(self):
        k, g, a, c = self.kgac
        has_deriv, no_deriv = self.derivs
        ks = self.ks
        u0 = self.u0
        dtc = float(self.dt)
        bt = self.butcher_tableau
        AA = bt.A
        BB = bt.b
        gsplit = g.subfunctions
        for i in range(self.num_stages):
            # compute the already-known part of the state in the
            # variational form
            g.assign(u0)
            for j in range(i):
                ksplit = ks[j].subfunctions
                for ind in has_deriv:
                    gbit = gsplit[ind]
                    kbit = ksplit[ind]
                    gbit += dtc * float(AA[i, j]) * kbit

            # update BC constants for the variational problem
            self.update_bc_constants(i, c)
            a.assign(AA[i, i])

            # solve new variational problem, stash the computed
            # stage value.

            # Note: implicitly uses solution value for stage i as
            # initial guess for stage i+1 and uses the last stage from
            # previous time step for stage 0 of the next one.  The
            # former is probably optimal, we hope for the best with
            # the latter.
            self.solver.solve()
            mysnes = self.solver.snes
            self.num_nonlinear_iterations += mysnes.getIterationNumber()
            self.num_linear_iterations += mysnes.getLinearSolveIterations()
            ks[i].assign(k)

            # Directly assign auxiliary variable at end of each stage
            for ind_ in no_deriv:
                u0bit = u0.subfunctions[ind_]
                kbit = ks[-1].subfunctions[ind_]
                u0bit.assign(kbit)

        # update the solution with now-computed stage values.
        for i in range(self.num_stages):
            for ind in has_deriv:
                u0bit = u0.subfunctions[ind]
                kbit = ks[i].subfunctions[ind]
                u0bit += dtc * float(BB[i]) * kbit

        self.num_steps += 1

    def solver_stats(self):
        return self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations
