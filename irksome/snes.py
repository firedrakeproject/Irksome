import copy
from firedrake import AuxiliaryOperatorSNES
from firedrake.dmhooks import get_appctx
from irksome.pc import ldu


class IRKAuxiliaryOperatorSNES(AuxiliaryOperatorSNES):
    """Base class that inherits from Firedrake's AuxiliaryOperatorSNES class and
    provides the preconditioning nonlinear form associated with an auxiliary
    Form and/or approximate Butcher matrix (which are provided by subclasses).
    """

    def getNewForm(self, snes, u0, test):
        """Derived classes can optionally provide an auxiliary Form.
        Must return Fnew, bcnew, unew."""
        raise NotImplementedError

    def getAtilde(self, A):
        """Derived classes produce a typically structured
        approximation to A."""
        raise NotImplementedError

    def form(self, snes, u, v):
        """Implements the interface for AuxiliaryOperatorSNES."""

        appctx = get_appctx(snes.dm).appctx
        stepper = appctx["stepper"]
        butcher = stepper.butcher_tableau
        F = stepper.F
        u0 = stepper.u0
        bcs = stepper.orig_bcs
        w = stepper.stages.copy(deepcopy=True)
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

        Fnew, bcnew = stepper.get_form_and_bcs(w, tableau=butcher, F=F)

        return Fnew, bcnew, w


class SDCLDSNES(IRKAuxiliaryOperatorSNES):
    """Implements SDC-type preconditioner using Atilde = DU where A=LDU. (Qdelta = Atilde)"""
    prefix = "sdc_"

    def getAtilde(self, A):
        L, D, U = ldu(A)
        return L @ D
