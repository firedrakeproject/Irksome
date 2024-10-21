import abc
import copy

import numpy
from firedrake import AuxiliaryOperatorPC, derivative
from ufl import replace

from irksome import getForm
from irksome.stage import getFormStage


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


class RanaBase(AuxiliaryOperatorPC):
    """Base class for methods out of Rana, Howle, Long, Meek, & Milestone.
    It inherits from Firedrake's AuxiliaryOperatorPC class and
    provides the preconditioning bilinear form associated with an
    approximation to the Butcher matrix (which is provided by
    subclasses)."""

    @abc.abstractmethod
    def getAtilde(self, A):
        """Derived classes produce a typically structured
        approximation to A."""
        pass

    def form(self, pc, test, trial):
        """Implements the interface for AuxiliaryOperatorPC."""
        appctx = self.get_appctx(pc)
        F = appctx["F"]
        butcher_tableau = appctx["butcher_tableau"]
        t = appctx["t"]
        dt = appctx["dt"]
        u0 = appctx["u0"]
        bcs = appctx["bcs"]
        stage_type = appctx.get("stage_type", None)
        bc_type = appctx.get("bc_type", None)
        splitting = appctx.get("splitting", None)
        nullspace = appctx.get("nullspace", None)

        # Make a modified Butcher tableau, probably with some kind
        # of sparser structure (e.g. LD part of LDU factorization)
        Atilde = self.getAtilde(butcher_tableau.A)
        butcher_new = copy.deepcopy(butcher_tableau)
        butcher_new.A = Atilde

        # which getForm do I need to get?

        if stage_type in ("deriv", None):
            Fnew, w, bcnew, bignsp = \
                getForm(F, butcher_new, t, dt, u0, bcs,
                        bc_type, splitting, nullspace)
        elif stage_type == "value":
            Fnew, _, w, bcnew, bignsp = \
                getFormStage(F, butcher_new, u0, t, dt, bcs,
                             splitting, nullspace)
        # Now we get the Jacobian for the modified system,
        # which becomes the auxiliary operator!
        test_old = Fnew.arguments()[0]
        a = replace(derivative(Fnew, w, du=trial),
                    {test_old: test})

        return a, bcnew


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


class IRKAuxiliaryOperatorPC(AuxiliaryOperatorPC):
    @abc.abstractmethod
    def getNewForm(self, pc, u0, test):
        pass

    def form(self, pc, test, trial):
        """Implements the interface for AuxiliaryOperatorPC."""
        appctx = self.get_appctx(pc)
        butcher_tableau = appctx["butcher_tableau"]
        oldF = appctx["F"]
        t = appctx["t"]
        dt = appctx["dt"]
        u0 = appctx["u0"]
        bcs = appctx["bcs"]
        stage_type = appctx.get("stage_type", None)
        bc_type = appctx.get("bc_type", None)
        splitting = appctx.get("splitting", None)
        nullspace = appctx.get("nullspace", None)
        v0 = oldF.arguments()[0]

        F, bcs = self.getNewForm(pc, u0, v0)
        # which getForm do I need to get?

        if stage_type in ("deriv", None):
            Fnew, w, bcnew, bignsp = \
                getForm(F, butcher_tableau, t, dt, u0, bcs,
                        bc_type, splitting, nullspace)
        elif stage_type == "value":
            Fnew, _, w, bcnew, bignsp = \
                getFormStage(F, butcher_tableau, u0, t, dt, bcs,
                             splitting, nullspace)
        # Now we get the Jacobian for the modified system,
        # which becomes the auxiliary operator!
        test_old = Fnew.arguments()[0]
        a = replace(derivative(Fnew, w, du=trial),
                    {test_old: test})

        return a, bcnew
