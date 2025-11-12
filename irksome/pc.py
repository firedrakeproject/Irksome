import copy

import numpy
from ufl import as_tensor
from .ButcherTableaux import CollocationButcherTableau
from .galerkin_stepper import ContinuousPetrovGalerkinTimeStepper
from .stage_derivative import getForm
from .stage_value import getFormStage
from .tools import replace, reshape, AI
from firedrake import AuxiliaryOperatorPC, Constant, derivative
from firedrake.dmhooks import get_appctx


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
        stepper = appctx["stepper"]
        galerkin = isinstance(stepper, ContinuousPetrovGalerkinTimeStepper)
        if galerkin:
            L = stepper.test_el
            butcher = CollocationButcherTableau(L, None)
        else:
            butcher = stepper.butcher_tableau
        F = stepper.F
        u0 = stepper.u0
        bcs = stepper.orig_bcs
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

        if galerkin:
            # Construct the equivalent IRK stage form
            if stepper.basis_type[0] == "value":
                Fnew, bcnew = getFormStage(F, butcher, stepper.t, stepper.dt, u0, w, bcs=bcs, splitting=AI)
            else:
                Fnew, bcnew = getForm(F, butcher, stepper.t, stepper.dt, u0, w, bcs=bcs, splitting=AI, bc_type="ODE")
            # GalerkinCollocation is equivalent to a Collocation IRK up to row scaling
            test, = Fnew.arguments()
            test_new = reshape(test, (-1, *v0.ufl_shape))
            for i, bi in enumerate(butcher.b):
                test_new[i] *= Constant(bi)
            test_new = as_tensor(test_new.reshape(test.ufl_shape))
            Fnew = replace(Fnew, {test: test_new})
        else:
            Fnew, bcnew = stepper.get_form_and_bcs(w, tableau=butcher, F=F)

        Jnew = derivative(Fnew, w, du=trial)

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


class NystromAuxiliaryOperatorPC(AuxiliaryOperatorPC):
    """Base class that inherits from Firedrake's AuxiliaryOperatorPC class and
    provides the preconditioning bilinear form associated with an auxiliary
    Form and/or approximate Nystrom matrices (which are provided by subclasses).
    """

    def getNewForm(self, pc, u0, ut0, test):
        """Derived classes can optionally provide an auxiliary Form."""
        raise NotImplementedError

    def getAtildes(self, A, Abar):
        """Derived classes produce a typically structured
        approximation to A and Abar."""
        raise NotImplementedError

    def form(self, pc, test, trial):
        """Implements the interface for AuxiliaryOperatorPC."""
        appctx = self.get_appctx(pc)
        stepper = appctx["stepper"]
        tableau = stepper.tableau
        F = stepper.F
        bcs = stepper.orig_bcs
        u0 = stepper.u0
        ut0 = stepper.ut0
        v0, = F.arguments()

        try:
            # use new Form if provided
            F, bcs = self.getNewForm(pc, u0, ut0, v0)
        except NotImplementedError:
            pass

        try:
            # use new ButcherTableau if provided
            Atilde, Abartilde = self.getAtildes(tableau.A, tableau.Abar)
            tableau = copy.deepcopy(tableau)
            tableau.A = Atilde
            tableau.Abar = Abartilde
        except NotImplementedError:
            pass

        # get stages
        ctx = get_appctx(pc.getDM())
        w = ctx._x

        Fnew, bcnew = stepper.get_form_and_bcs(w, tableau=tableau, F=F)
        Jnew = derivative(Fnew, w, du=trial)

        return Jnew, bcnew


class ClinesBase(NystromAuxiliaryOperatorPC):
    """Base class for methods out of Clines/Howle/Long."""
    pass


class ClinesLD(ClinesBase):
    """Implements Clines-type preconditioner using Atilde = LD where A=LDU."""
    def getAtildes(self, A, Abar):
        L, D, _ = ldu(A)
        Atilde = L @ D
        try:
            Lbar, Dbar, _ = ldu(Abar)
        except AssertionError:
            raise ValueError(
                "ClinesLD preconditioner failed for for this tableau.  Please try again with GaussLegendre or RadauIIA methods")
        Abartilde = Lbar @ Dbar
        return Atilde, Abartilde
