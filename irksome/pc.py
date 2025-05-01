import copy

import numpy
from irksome.discontinuous_galerkin_stepper import DiscontinuousGalerkinTimeStepper, getElement
from irksome.tools import vecconst
from firedrake import Function, AuxiliaryOperatorPC, TwoLevelPC, derivative
from firedrake.dmhooks import get_appctx
from firedrake.petsc import PETSc


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



class Interpolator:
    def __init__(self, V, Vbig, source, target):
        self.u = Function(V)
        self.ubig = Function(Vbig)

        primal = source.get_nodal_basis()
        dual = target.get_dual_set()
        A = dual.to_riesz(primal)
        B = primal.get_coeffs()
        b = (A @ B).reshape((-1,))
        self.bs = vecconst(b)

    def mult(self, mat, x, y):
        with self.u.dat.vec_wo as xv:
            x.copy(xv)

        num_fields = len(self.u.subfunctions)
        for i in range(num_fields):
            xi = self.u.subfunctions[i]
            yi = numpy.array(self.ubig.subfunctions[i::num_fields], dtype=object)
            for j in range(len(self.bs)):
                yi[j].assign(xi * self.bs[j])

        with self.ubig.dat.vec_ro as yv:
            yv.copy(y)

    def multTranspose(self, mat, x, y):
        with self.ubig.dat.vec_wo as xv:
            x.copy(xv)

        num_fields = len(self.u.subfunctions)
        for i in range(num_fields):
            yi = self.u.subfunctions[i]
            xi = numpy.array(self.ubig.subfunctions[i::num_fields], dtype=object)
            yi.assign(numpy.dot(xi, self.bs))

        with self.u.dat.vec_ro as yv:
            yv.copy(y)


class LowOrderInTimePC(TwoLevelPC):

    _prefix = "pmg_"

    def coarsen(self, pc):
        appctx = self.get_appctx(pc)
        stepper = appctx["stepper"]

        is_dg = isinstance(stepper, DiscontinuousGalerkinTimeStepper)
        order = 0 if is_dg else 1

        w = stepper.u0
        F, bcs = stepper.get_form_and_bcs(w, order=order)
        a = derivative(F, w)

        elbig = stepper.el if is_dg else stepper.trial_el
        el = type(elbig)(elbig.ref_el, order)
        el = getElement(stepper.basis_type, order)

        V = stepper.u0.function_space()
        Vbig = stepper.stages.function_space()

        shell = Interpolator(V, Vbig, el, elbig)
        sizes = (Vbig.dof_dset.layout_vec.getSizes(),
                 V.dof_dset.layout_vec.getSizes())
        I = PETSc.Mat().createPython(sizes, shell, comm=V._comm)
        I.setUp()
        return a, bcs, I
