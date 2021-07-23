import abc
import copy
from firedrake import AuxiliaryOperatorPC, derivative
from irksome import getForm
import numpy
from ufl import replace


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
    @abc.abstractmethod
    def getAtilde(self, A):
        pass

    def form(self, pc, test, trial):
        # Fish out information from the appctx (which TimeStepper
        # puts in when it sets up the NLVP)
        appctx = self.get_appctx(pc)
        F = appctx["F"]
        butcher_tableau = appctx["butcher_tableau"]
        t = appctx["t"]
        dt = appctx["dt"]
        u0 = appctx["u0"]
        bcs = appctx["bcs"]
        bc_type = appctx["bc_type"]
        splitting = appctx["splitting"]

        # Make a modified Butcher tableau, probably with some kind
        # of sparser structure (e.g. LD part of LDU factorization)
        Atilde = self.getAtilde(butcher_tableau.A)
        butcher_new = copy.deepcopy(butcher_tableau)
        butcher_new.A = Atilde

        # Get the UFL for the system with the modified Butcher tableau
        Fnew, w, bcnew, _ = getForm(F, butcher_new, t, dt, u0, bcs,
                                    bc_type, splitting)

        # Now we get the Jacobian for the modified system,
        # which becomes the auxiliary operator!
        test_old = Fnew.arguments()[0]
        a = replace(derivative(Fnew, w, du=trial),
                    {test_old: test})

        return a, bcnew


class RanaLD(RanaBase):
    def getAtilde(self, A):
        L, D, U = ldu(A)
        return L @ D


class RanaDU(RanaBase):
    def getAtilde(self, A):
        L, D, U = ldu(A)
        return D @ U
