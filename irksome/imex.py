from .stage import getFormStage
from .tools import replace, AI, IA
from .ButcherTableaux import RadauIIA
from firedrake import (Constant, Function, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, TestFunction)
import FIAT
import numpy as np
from ufl.classes import Zero
from .stage import getBits


def explicit_coeffs(k):
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
    method."""
    v = Fexp.arguments()[0]
    V = v.function_space()
    Vbig = UU.function_space()
    VV = TestFunction(Vbig)

    num_stages = butch.num_stages
    num_fields = len(V)
    vc = np.vectorize(Constant)
    Aexp = explicit_coeffs(num_stages)
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
        # TODO: IA is longer to do since
        # you have a diagonal contribution to the propagator
        # and a dense one to the iterator with coefficients
        # equal to inverse of A onto the explicit coefficients.
        raise NotImplementedError("Unsupported splitting type")
    else:
        raise NotImplementedError("Unsupported splitting type")

    return Fit, Fprop


class RadauIIAIMEXMethod():
    def __init__(self, F, Fexp, butcher_tableau,
                 t, dt, u0, bcs=None,
                 it_solver_parameters=None,
                 prop_solver_parameters=None,
                 splitting=AI,
                 appctx=None,
                 nullspace=None):
        assert isinstance(butcher_tableau, RadauIIA)

        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = len(butcher_tableau.b)
        self.butcher_tableau = butcher_tableau

        # Since this assumes RadauIIA, we drop
        # the update information on the floor.
        Fbig, _, UU, bigBCs, gblah, nsp = getFormStage(
            F, butcher_tableau, u0, t, dt, bcs,
            splitting=splitting)

        self.UU = UU
        self.UU_old = UU_old = Function(UU.function_space())
        self.UU_old_split = UU_old.split()
        self.bigBCs = bigBCs
        self.bcdat = gblah

        Fit, Fprop = getFormExplicit(
            Fexp, butcher_tableau, u0, UU_old, t, dt, splitting)

        self.itprob = NonlinearVariationalProblem(
            Fbig + Fit, UU, bcs=bigBCs)
        self.propprob = NonlinearVariationalProblem(
            Fbig + Fprop, UU, bcs=bigBCs)
        self.it_solver = NonlinearVariationalSolver(
            self.itprob, solver_parameters=it_solver_parameters)
        self.prop_solver = NonlinearVariationalSolver(
            self.propprob, solver_parameters=prop_solver_parameters)

        for uolddat in self.UU_old.dat:
            uolddat.data[:] = u0.dat.data_ro[:]

    def iterate(self):
        self.it_solver.solve()
        for uod, uud in zip(self.UU_old.dat, self.UU.dat):
            uod.data[:] = uud.data_ro[:]

    def advance(self):
        self.u0.assign(self.UU_old_split[-1])
        for gdat, gcur, gmethod in self.bcdat:
            gmethod(gdat, gcur)
        self.prop_solver.solve()
        for uod, uud in zip(self.UU_old.dat, self.UU.dat):
            uod.data[:] = uud.data_ro[:]
