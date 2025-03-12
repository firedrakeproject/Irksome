from .base_time_stepper import StageCoupledTimeStepper
from .bcs import BCStageData, bc2space
from .deriv import Dt, TimeDerivative, expand_time_derivatives
from .tools import replace, vecconst
from firedrake import TestFunction, as_ufl
import numpy
from ufl import zero


class NystromTableau:
    def __init__(self, A, b, c, Abar, bbar, order):
        self.A = A
        self.b = b
        self.c = c
        self.Abar = Abar
        self.bbar = bbar
        self.order = order

    @property
    def num_stages(self):
        return len(self.b)

    @property
    def is_explicit(self):
        return (numpy.allclose(numpy.triu(self.Abar), 0)
                and numpy.allclose(numpy.triu(self.A), 0))

    @property
    def is_diagonally_implicit(self):
        return (numpy.allclose(numpy.triu(self.Abar, 1), 0)
                and numpy.allclose(numpy.triu(self.A, 1), 0))

    @property
    def is_implicit(self):
        return not self.is_explicit

    @property
    def is_fully_implicit(self):
        return self.is_implicit and not self.is_diagonally_implicit

    def __str__(self):
        return str(self.__class__).split(".")[-1][:-2]+"()"


def butcher_to_nystrom(butch):
    A = butch.A
    b = butch.b
    return NystromTableau(A, b, butch.c, A @ A, A.T @ b, butch.order)


# Not all Nystrom methods come from RK
class ClassicNystrom4Tableau(NystromTableau):
    def __init__(self):
        A = numpy.zeros((4, 4))
        Abar = numpy.zeros((4, 4))

        A[1, 0] = 0.5
        A[2, 1] = 0.5
        A[3, 2] = 1

        Abar[1, 0] = 1./8
        Abar[2, 0] = 1./8
        Abar[3, 2] = 1.0

        b = numpy.array([1, 2, 2, 1]) / 6.
        bbar = numpy.array([1, 1, 1, 0]) / 6.

        c = numpy.array([0, 0.5, 0.5, 1])

        super().__init__(A, b, c, Abar, bbar, 4)


def getFormNystrom(F, tableau, t, dt, u0, ut0, stages,
                   bcs=None, bc_type=None):
    if bc_type is None:
        bc_type = "DAE"

    # preprocess time derivatives
    F = expand_time_derivatives(F, t=t, timedep_coeffs=(u0,))
    v = F.arguments()[0]
    V = v.function_space()
    assert V == u0.function_space()

    A = vecconst(tableau.A)
    Abar = vecconst(tableau.Abar)
    c = vecconst(tableau.c)

    num_stages = tableau.num_stages
    Vbig = stages.function_space()
    test = TestFunction(Vbig)

    v_np = numpy.reshape(test, (num_stages, *u0.ufl_shape))
    k_np = numpy.reshape(stages, (num_stages, *u0.ufl_shape))

    Ak = A @ k_np
    Abark = Abar @ k_np

    dtu = TimeDerivative(u0)
    dt2u = TimeDerivative(dtu)

    Fnew = zero()

    for i in range(num_stages):
        repl = {t: t + c[i] * dt,
                v: v_np[i],
                u0: u0 + ut0 * (c[i] * dt) + Abark[i] * dt**2,
                dtu: ut0 + Ak[i] * dt,
                dt2u: k_np[i]}
        Fnew += replace(F, repl)

    if bcs is None:
        bcs = []
    if bc_type == "ODE":
        def bc2gcur(bc, i):
            gorig = as_ufl(bc._original_arg)
            gfoo = expand_time_derivatives(Dt(gorig, 2), t=t, timedep_coeffs=(u0,))
            return replace(gfoo, {t: t + c[i] * dt})

    elif bc_type == "DAE":
        try:
            bA1inv = numpy.linalg.inv(tableau.Abar)
            A1inv = vecconst(bA1inv)
        except numpy.linalg.LinAlgError:
            raise NotImplementedError("Can't have DAE BC's for this method")

        def bc2gcur(bc, i):
            gorig = as_ufl(bc._original_arg)
            ucur = bc2space(bc, u0)
            utcur = bc2space(bc, ut0)
            gcur = (1/dt**2) * sum((replace(gorig, {t: t + c[j]*dt}) - ucur - utcur * (dt * c[j])) * A1inv[i, j]
                                   for j in range(num_stages))
            return gcur

    elif bc_type == "dDAE":
        if tableau.is_explicit:
            try:
                AAb = numpy.vstack((tableau.A, tableau.b))
                AAb = AAb[1:]
                bA1inv = numpy.linalg.inv(AAb)
                A1inv = vecconst(bA1inv)
                c_one = numpy.append(tableau.c, 1.0)
                c_ddae = vecconst(c_one[1:])
            except numpy.linalg.LinAlgError:
                raise NotImplementedError("Can't have derivative DAE BC's for this method")
        else:
            try:
                bA1inv = numpy.linalg.inv(tableau.A)
                A1inv = vecconst(bA1inv)
                c_ddae = vecconst(tableau.c)
            except numpy.linalg.LinAlgError:
                raise NotImplementedError("Can't have derivative DAE BC's for this method")

        def bc2gcur(bc, i):
            gorig = as_ufl(bc._original_arg)
            gfoo = expand_time_derivatives(Dt(gorig, 1), t=t, timedep_coeffs=(u0,))
            utcur = bc2space(bc, ut0)
            gcur = (1/dt) * sum((replace(gfoo, {t: t + c_ddae[j]*dt}) - utcur) * A1inv[i, j]
                                for j in range(num_stages))
            return gcur

    else:
        raise ValueError(f"Unrecognized BC type: {bc_type}")

    bcnew = []
    for bc in bcs:
        for i in range(num_stages):
            gcur = bc2gcur(bc, i)
            bcnew.append(BCStageData(bc, gcur, u0, stages, i))

    return Fnew, bcnew


class StageDerivativeNystromTimeStepper(StageCoupledTimeStepper):
    def __init__(self, F, tableau, t, dt, u0, ut0,
                 bcs=None, solver_parameters=None,
                 appctx=None, nullspace=None,
                 bc_type="DAE"):
        self.ut0 = ut0
        if not isinstance(tableau, NystromTableau):
            tableau = butcher_to_nystrom(tableau)

        self.tableau = tableau

        super().__init__(F, t, dt, u0,
                         tableau.num_stages, bcs=bcs,
                         solver_parameters=solver_parameters,
                         appctx=appctx, nullspace=nullspace,
                         bc_type=bc_type)

        self.updateb = vecconst(tableau.b)
        self.updatebbar = vecconst(tableau.bbar)
        self.num_fields = len(u0.function_space())

    def _update(self):
        b = self.updateb
        bbar = self.updatebbar
        ns = self.tableau.num_stages
        dt = self.dt
        nf = self.num_fields

        # Note: order matters here.  derivative update doesn't
        # depend on old solution value.
        kp = self.stages.subfunctions
        for i, (u0bit, ut0bit) in enumerate(zip(self.u0.subfunctions,
                                                self.ut0.subfunctions)):
            u0bit += (ut0bit * dt
                      + sum(kp[nf * s + i] * (bbar[s] * dt**2)
                            for s in range(ns)))
            ut0bit += sum(kp[nf * s + i] * (b[s] * dt) for s in range(ns))

    def get_form_and_bcs(self, stages, tableau=None):
        if tableau is None:
            tableau = self.tableau
        return getFormNystrom(self.F, tableau, self.t,
                              self.dt, self.u0, self.ut0,
                              stages,
                              bcs=self.orig_bcs,
                              bc_type=self.bc_type)
