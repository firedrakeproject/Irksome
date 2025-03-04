from .base_time_stepper import StageCoupledTimeStepper
from .bcs import BCStageData, bc2space
from .deriv import TimeDerivative
from .tools import component_replace, replace, vecconst
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


def butcher_to_nystrom(butch):
    A = butch.A
    b = butch.b
    return NystromTableau(A, b, butch.c, A @ A, A.T @ b, butch.order)


def getFormNystrom(F, tableau, t, dt, u0, ut0, stages,
                   bcs=None, bc_type=None):
    if bc_type is None:
        bc_type = "DAE"
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
                u0: u0 + c[i] * dt * ut0 + dt**2 * Abark[i],
                dtu: ut0 + dt * Ak[i],
                dt2u: k_np[i]}
        Fnew += component_replace(F, repl)

    if bcs is None:
        bcs = []
    if bc_type != "DAE":
        raise ValueError(f"Unrecognized BC type: {bc_type}")
    else:
        try:
            bA1inv = numpy.linalg.inv(tableau.A)
            A1inv = vecconst(bA1inv)
        except numpy.linalg.LinAlgError:
            raise NotImplementedError("Can't have DAE BC's for this method")

        def bc2gcur(bc, i):
            gorig = as_ufl(bc._original_arg)
            ucur = bc2space(bc, u0)
            gcur = (1/dt) * sum((replace(gorig, {t: t + c[j]*dt}) - ucur) * A1inv[i, j]
                                for j in range(num_stages))
            return gcur

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
                              self.stages,
                              bcs=self.orig_bcs,
                              bc_type=self.bc_type)
