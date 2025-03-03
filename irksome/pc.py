import copy

import numpy
from firedrake import AuxiliaryOperatorPC, derivative
from firedrake.dmhooks import get_appctx
from ufl import replace

from irksome.stage_derivative import getForm
from irksome.stage_value import getFormStage


# Oddly, we can't turn pivoting off in scipy?
def ldu(A):
    m = A.shape[0]
    assert m == A.shape[1]

    L = numpy.eye(m)
    U = numpy.copy(A)
    D = numpy.zeros((m, m))

    for k in range(m):
        for i in range(k+1, m):
            alpha = U[i, k] / U[k, k]
            U[i, :] -= alpha * U[k, :]
            L[i, k] = alpha

    assert numpy.allclose(L @ U, A)

    for k in range(m):
        D[k, k] = U[k, k]
        U[k, k:] /= D[k, k]

    assert numpy.allclose(L @ D @ U, A)

    return L, D, U


class IRKAuxiliaryOperatorPC(AuxiliaryOperatorPC):
    """Base class that inherits from Firedrake's AuxiliaryOperatorPC class and
    provides the preconditioning bilinear form associated with an auxiliary
    Form and/or approximate Butcher matrix (which are provided by subclasses).
    """

    def getNewForm(self, pc, u0, test):
        """Derived classes can optionally provide an auxiliary Form."""
        raise NotImplementedError

    def getAtilde(self, A):
        """Derived classes produce a typically structured
        approximation to A."""
        raise NotImplementedError

    def form(self, pc, test, trial):
        """Implements the interface for AuxiliaryOperatorPC."""
        appctx = self.get_appctx(pc)
        butcher = appctx["butcher_tableau"]
        F = appctx["F"]
        t = appctx["t"]
        dt = appctx["dt"]
        u0 = appctx["u0"]
        bcs = appctx["bcs"]
        stage_type = appctx.get("stage_type", None)
        bc_type = appctx.get("bc_type", None)
        splitting = appctx.get("splitting", None)
        v0, = F.arguments()

        try:
            # use new Form if provided
            F, bcs = self.getNewForm(pc, u0, v0)
        except NotImplementedError:
            pass

        try:
            # use new ButcherTableau if provided
            Atilde = self.getAtilde(butcher.A)
            butcher = copy.deepcopy(butcher)
            butcher.A = Atilde
        except NotImplementedError:
            pass

        # get stages
        ctx = get_appctx(pc.getDM())
        w = ctx._x

        # which getForm do I need to get?
        if stage_type in ("deriv", None):
            Fnew, bcnew = getForm(F, butcher, t, dt, u0, w, bcs, bc_type, splitting)
        elif stage_type == "value":
            Fnew, bcnew = getFormStage(F, butcher, t, dt, u0, w, bcs, splitting)

        # Now we get the Jacobian for the modified system,
        # which becomes the auxiliary operator!
        test_old = Fnew.arguments()[0]
        Jnew = replace(derivative(Fnew, w, du=trial),
                       {test_old: test})

        return Jnew, bcnew


class RanaBase(IRKAuxiliaryOperatorPC):
    """Base class for methods out of Rana, Howle, Long, Meek, & Milestone."""
    pass


class RanaLD(RanaBase):
    """Implements Rana-type preconditioner using Atilde = LD where A=LDU."""
    def getAtilde(self, A):
        L, D, U = ldu(A)
        return L @ D


class RanaDU(RanaBase):
    """Implements Rana-type preconditioner using Atilde = DU where A=LDU."""
    def getAtilde(self, A):
        L, D, U = ldu(A)
        return D @ U
