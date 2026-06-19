import copy

import numpy
from .labeling import as_form
from firedrake import AuxiliaryOperatorPC, AuxiliaryOperatorSNES, derivative
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
        butcher = stepper.butcher_tableau
        F = as_form(stepper.F)
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

        Fnew, bcnew = stepper.get_form_and_bcs(w, tableau=butcher, F=F)
        Jnew = derivative(Fnew, w, du=trial)

        return Jnew, bcnew


class IRKAuxiliaryOperatorSNES(AuxiliaryOperatorSNES):
    """Base class that inherits from Firedrake's :class:`AuxiliaryOperatorSNES`
    and provides the nonlinear form associated with an auxiliary Form and/or
    approximate Butcher matrix (which are provided by subclasses). This is the
    nonlinear analogue of :class:`IRKAuxiliaryOperatorPC`.

    Options for the inner solve are specified using the ``"aux_"`` prefix, as
    for :class:`AuxiliaryOperatorSNES`.
    """

    def getNewForm(self, snes, u0, test):
        """Derived classes can optionally provide an auxiliary semidiscrete
        Form, expressed in terms of the current state ``u0`` and a ``test``
        function over the same (single-stage) function space."""
        raise NotImplementedError

    def getAtilde(self, A):
        """Derived classes produce a typically structured
        approximation to A."""
        raise NotImplementedError

    def form(self, snes, w0, w, test):
        """Implements the interface for AuxiliaryOperatorSNES.

        :arg snes: the PETSc SNES object.
        :arg w0: the current iterate of the stages (unused by default, but
            available to subclasses that wish to lag terms).
        :arg w: the stages to be solved for at the next iterate; the auxiliary
            residual is built in terms of this Function.
        :arg test: the test function over the stage space (unused; the time
            stepper builds its own test function).
        """
        appctx = self.get_appctx(snes)
        stepper = appctx["stepper"]
        butcher = stepper.butcher_tableau
        F = as_form(stepper.F)
        u0 = stepper.u0
        bcs = stepper.orig_bcs
        v0, = F.arguments()

        try:
            # use new Form if provided
            F, bcs = self.getNewForm(snes, u0, v0)
        except NotImplementedError:
            pass

        try:
            # use new ButcherTableau if provided
            Atilde = self.getAtilde(butcher.A)
            butcher = copy.deepcopy(butcher)
            butcher.A = Atilde
        except NotImplementedError:
            pass

        Fnew, bcnew = stepper.get_form_and_bcs(w, tableau=butcher, F=F, bcs=bcs)

        return Fnew, bcnew


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
