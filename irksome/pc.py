import copy
from contextlib import contextmanager, nullcontext

import numpy
from scipy.linalg import solve_sylvester

from .labeling import as_form
from .nystrom_stepper import StageDerivativeNystromTimeStepper
from .tableaux.ButcherTableaux import ButcherTableau, CollocationButcherTableau
from .scheme import GalerkinCollocationScheme, DiscontinuousGalerkinCollocationScheme
from .discontinuous_galerkin_stepper import getElement
from .galerkin_stepper import getTestElement

from firedrake import AuxiliaryOperatorPC, PCBase, derivative
from firedrake.dmhooks import get_appctx
from firedrake.petsc import PETSc

try:
    from firedrake import AuxiliaryOperatorSNES
    has_auxiliary_operator_snes = True
except ImportError:
    has_auxiliary_operator_snes = False

    class AuxiliaryOperatorSNES:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "IRKAuxiliaryOperatorSNES requires firedrake.AuxiliaryOperatorSNES,"
                " which this version of Firedrake does not provide; please upgrade."
            )


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


def as_butcher_tableau(scheme):
    """Convert a scheme to its ButcherTableau equivalent."""
    if isinstance(scheme, ButcherTableau):
        return scheme

    if isinstance(scheme, GalerkinCollocationScheme):
        basis_type = scheme.basis_type
        if isinstance(basis_type, tuple):
            basis_type = basis_type[1]
        element = getTestElement(basis_type, scheme.order-1)
    elif isinstance(scheme, DiscontinuousGalerkinCollocationScheme):
        element = getElement(scheme.basis_type, scheme.order)
    else:
        raise TypeError(f"Cannot convert a {type(scheme).__name__} into a ButcherTableau.")

    return CollocationButcherTableau(element, scheme.order)


def RanaLDScheme(scheme):
    """ButcherTableau for preconditioning with Atilde = LD where A=LDU."""
    butcher = as_butcher_tableau(scheme)
    L, D, U = ldu(butcher.A)
    return butcher.reconstruct(A=L @ D)


def RanaDUScheme(scheme):
    """ButcherTableau for preconditioning with Atilde = DU where A=LDU."""
    butcher = as_butcher_tableau(scheme)
    L, D, U = ldu(butcher.A)
    return butcher.reconstruct(A=D @ U)


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

        u0 = stepper.u0
        bcs = stepper.orig_bcs

        try:
            # use new Form if provided
            F = as_form(stepper.F)
            v0, = F.arguments()
            F, bcs = self.getNewForm(pc, u0, v0)
        except NotImplementedError:
            F = stepper.Jp or stepper.J or stepper.F
            F = as_form(F)

        try:
            # use new ButcherTableau if provided
            Atilde = self.getAtilde(butcher.A)
            butcher = butcher.reconstruct(A=Atilde)
        except NotImplementedError:
            pass

        # get stages
        ctx = get_appctx(pc.getDM())
        w = ctx._x

        Fnew, bcnew = stepper.get_form_and_bcs(w, tableau=butcher, F=F)
        Jnew = derivative(Fnew, w, du=trial)
        return Jnew, bcnew


class IRKAuxiliaryOperatorSNES(AuxiliaryOperatorSNES):
    """Base class that inherits from Firedrake's auxiliary operator SNES
    and provides the nonlinear form associated with an auxiliary Form and/or
    approximate Butcher matrix (which are provided by subclasses). This is the
    nonlinear analogue of :class:`IRKAuxiliaryOperatorPC`.

    Options for the inner solve are specified using the ``"aux_"`` prefix, as
    for the base auxiliary operator SNES.
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
        bcs = stepper.orig_bcs
        u0 = stepper.u0

        if not isinstance(stepper, StageDerivativeNystromTimeStepper):
            raise TypeError("Expecting a Nystrom stepper")

        tableau = stepper.tableau
        ut0 = stepper.ut0

        try:
            # use new Form if provided
            F = stepper.F
            v0, = F.arguments()
            F, bcs = self.getNewForm(pc, u0, ut0, v0)
        except NotImplementedError:
            F = stepper.Jp or stepper.J or stepper.F

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


# Preconditioning by the solution of a Sylvester equation
#
# With a single Jacobian Jbar for all stages, the stage Jacobian of an
# s-stage method is I x M + dt A x Jbar. Premultiplied by (dt A1)^{-1} x I
# (A = A2 A1 is the splitting of the stepper), it is the Sylvester operator
# X -> K X + M X S^T with S = (dt A)^{-1} - sigma I and K = Jbar + sigma M,
# where X is the n x s matrix of the stage blocks. The equation is solved
# with the extended Krylov subspace method (EKSM), which needs only solves
# with K (a backward Euler matrix) and with M.


def _get_stages(x, vecs):
    """Copy the stage blocks of the stage vector x into the single-stage
    Vecs vecs. The local part of x is [stage 0, stage 1, ...]."""
    n = vecs[0].getLocalSize()
    arr = x.getArray(readonly=True)
    for i, v in enumerate(vecs):
        v.array[:] = arr[i * n:(i + 1) * n]


def _set_stages(vecs, x):
    """Inverse of _get_stages."""
    n = vecs[0].getLocalSize()
    with x as arr:
        for i, v in enumerate(vecs):
            arr[i * n:(i + 1) * n] = v.array_r


class _OrthonormalBasis:
    """A list of orthonormal Vecs."""

    def __init__(self):
        self.vectors = []

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, i):
        return self.vectors[i]

    def append(self, w):
        """Orthonormalize a copy of w against the basis (classical
        Gram-Schmidt, twice) and append it.

        :returns: the coefficients h of w in the basis and the norm of the
            orthogonal part.
        """
        v = w.copy()
        h = numpy.zeros(len(self.vectors))
        if self.vectors:
            for _ in range(2):
                hk = numpy.asarray(v.mDot(self.vectors)).real
                v.maxpy(-hk, self.vectors)
                h += hk
        norm = v.normalize()
        self.vectors.append(v)
        return h, norm

    def combine(self, y, result):
        """result = sum_j y[j] V[j]."""
        result.zeroEntries()
        result.maxpy(y, self.vectors[:len(y)])
        return result


def _extend_basis(op, V, vecs, offset, H, L=None, block_op=None):
    """Append op(v) for v in vecs to V. The coefficients of op(v) in the
    basis go to the columns offset, offset + 1, ... of H, and L gets ones on
    the diagonal of those columns. block_op, if given, maps the list vecs to
    the list of op(v) in one call."""
    if block_op is not None:
        ws = block_op(vecs)
    else:
        work = vecs[0].duplicate()
    for j, v in enumerate(vecs):
        if block_op is not None:
            w = ws[j]
        else:
            work.zeroEntries()
            op(v, work)
            w = work
        h, norm = V.append(w)
        H[:len(h), offset + j] = h
        H[len(h), offset + j] = norm
        if L is not None:
            L[offset + j, offset + j] = 1


class SylvesterOperator:
    """Python Mat context for the operator x -> vec(K X + M X S^T), where
    X is the n x s matrix of the stage blocks of x.

    The mass matrix is M = D M1 D, with D = diag(I, beta I), where beta
    scales the entries alg_dofs of a single-stage vector. M1 is the matrix
    that is assembled and solved with, so a change of beta needs no new
    assembly and no new solver set-up.

    :arg K: the single-stage matrix K.
    :arg M1: the single-stage matrix M1.
    :arg S: the (s, s) array S.
    :arg dm: optional single-stage DM; it is attached to the inner solvers
        of :class:`SylvesterEKSM` with the application contexts Kctx and
        Mctx, so that DM-aware preconditioners (e.g. geometric multigrid)
        can be used for the solves with K and M1.
    :arg alg_dofs: the local indices that D scales (None: D = I).
    """

    def __init__(self, K, M1, S, dm=None, Kctx=None, Mctx=None, alg_dofs=None):
        self.K = K
        self.M = M1
        self.S = numpy.asarray(S, dtype=float)
        self.dm = dm
        self.Kctx = Kctx
        self.Mctx = Mctx
        s = self.S.shape[0]
        self._x = [K.createVecRight() for _ in range(s)]
        self._Kx = [K.createVecLeft() for _ in range(s)]
        self._Mx = [K.createVecLeft() for _ in range(s)]
        self._tmp = K.createVecRight()
        self.alg_dofs = alg_dofs
        self.D = self.Dinv = None
        if alg_dofs is not None:
            self.D = K.createVecRight()
            self.Dinv = K.createVecRight()
            self.set_beta(1.0)

    def set_beta(self, beta):
        for d, value in ((self.D, beta), (self.Dinv, 1 / beta)):
            d.set(1.0)
            with d as arr:
                arr[self.alg_dofs] = value

    def scale(self, v, d):
        """v <- d v (pointwise), with d = D or D^{-1}."""
        if d is not None:
            v.pointwiseMult(v, d)

    def scale_dense(self, X, d):
        """Scale the rows of the dense matrix X by d."""
        if d is not None:
            X.diagonalScale(L=d)

    def Mmult(self, v, y):
        """y = M v = D M1 D v"""
        v.copy(self._tmp)
        self.scale(self._tmp, self.D)
        self.M.mult(self._tmp, y)
        self.scale(y, self.D)

    def mult(self, mat, x, y):
        _get_stages(x, self._x)
        for xi, Kxi, Mxi in zip(self._x, self._Kx, self._Mx):
            self.K.mult(xi, Kxi)
            self.Mmult(xi, Mxi)
        for i, Kxi in enumerate(self._Kx):
            Kxi.maxpy(self.S[i], self._Mx)
        _set_stages(self._Kx, y)


@contextmanager
def _temporary_options(prefix, values):
    """Set options in the global database for the duration of the
    context, restoring the previous values afterwards."""
    opts = PETSc.Options()
    old = {}
    for name, value in values.items():
        key = prefix + name
        old[key] = opts.getString(key) if opts.hasName(key) else None
        opts[key] = value
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                opts.delValue(key)
            else:
                opts[key] = value


# KSP type -> (HPDDM pseudo-block type, HPDDM variant)
_PSEUDO_BLOCK_KSP = {
    PETSc.KSP.Type.CG: ("cg", None),
    PETSc.KSP.Type.GMRES: ("gmres", None),
    PETSc.KSP.Type.FGMRES: ("gmres", "flexible"),
}


def _vecs_to_dense(vecs, D):
    for j, v in enumerate(vecs):
        c = D.getDenseColumnVec(j, mode="w")
        v.copy(c)
        D.restoreDenseColumnVec(j, mode="w", V=c)


def _dense_to_vecs(D, vecs=None):
    ncols = D.getSize()[1]
    if vecs is None:
        vecs = [None] * ncols
    for j in range(ncols):
        c = D.getDenseColumnVec(j, mode="r")
        if vecs[j] is None:
            vecs[j] = c.copy()
        else:
            c.copy(vecs[j])
        D.restoreDenseColumnVec(j, mode="r", V=c)
    return vecs


class _HookOwner:
    """Somewhere for firedrake.dmhooks.add_hooks to store set-up hooks.
    Each inner solver needs its own."""


class SylvesterEKSM:
    """Python KSP context: the extended Krylov subspace method (EKSM) for
    the Sylvester equation K X + M X S^T = B, with the operator given by a
    :class:`SylvesterOperator`.

    EKSM builds an orthonormal basis of the block extended Krylov subspace
    span{B', Kh^{-1} B', Kh B', Kh^{-2} B', ...} with Kh = M^{-1} K and
    B' = M^{-1} B, and solves the projected Sylvester equation with the
    Bartels-Stewart method (:func:`scipy.linalg.solve_sylvester`). Each
    iteration needs one block solve with K and one with M. The initial guess is zero, and the method keeps the whole
    basis, so the maximum number of iterations sets its memory. The
    convergence test uses the residual of the Sylvester equation, or the
    residual given by ``residual_fn`` if it is set.

    The solves with K and M use the options prefixes ``K_`` and ``M_``
    after the prefix of the KSP. Both default to a sparse direct solver.

    :kwarg fuse: apply the solves with K and M to a whole block of vectors
        at once (needs PETSc with HPDDM for iterative inner solvers).
    :kwarg adaptive_tol: increase the relative tolerance of the K solves as
        the residual decreases (for iterative K solves).
    """

    # stop when the residual given by residual_fn has not decreased for
    # this many iterations, and return the best iterate
    stagnation_its = 3
    # largest relative tolerance of the K solves with adaptive_tol
    max_K_rtol = 0.1
    # singular values of B' below this times the largest are round-off
    rank_tol = 1e3 * numpy.finfo(float).eps

    def __init__(self, fuse=False, adaptive_tol=True):
        self.fuse = fuse
        self.adaptive_tol = adaptive_tol
        # optional callable: residual_fn(xvec) -> (relative, absolute)
        # residual norms of a candidate solution xvec
        self.residual_fn = None
        self.rank = None

    def _inner(self, which, save=True):
        """Context for the set-up or application of the inner solver
        `which` ("K" or "M"): it pushes its Firedrake application context."""
        if self._dm is None:
            return nullcontext()
        from firedrake import dmhooks
        return dmhooks.add_hooks(self._dm, self._hook_owners[which],
                                 appctx=self._appctxs[which], save=save)

    def _create_inner_ksp(self, ksp, op, which):
        inner = PETSc.KSP().create(comm=ksp.comm)
        inner.setOperators(op, op)
        inner.setOptionsPrefix((ksp.getOptionsPrefix() or "") + which + "_")
        inner.setType(PETSc.KSP.Type.PREONLY)
        inner.getPC().setType(PETSc.PC.Type.LU)
        if PETSc.Sys.hasExternalPackage("mumps"):
            inner.getPC().setFactorSolverType("mumps")
        if self._dm is not None:
            inner.setDM(self._dm)
            # the operator is ours: the DM is only used by the PC
            inner.setDMActive(PETSc.KSP.DMActive.ALL, False)
        inner.incrementTabLevel(1, parent=ksp)
        return inner

    def _fuse_inner_ksp(self, inner):
        """Switch an inner KSP to its HPDDM pseudo-block equivalent."""
        ksp_type = inner.getType()
        if ksp_type in (PETSc.KSP.Type.HPDDM, PETSc.KSP.Type.PREONLY):
            return
        if ksp_type not in _PSEUDO_BLOCK_KSP:
            raise ValueError(f"SylvesterEKSM: cannot fuse the solves of a"
                             f" KSP of type {ksp_type}")
        if not PETSc.Sys.hasExternalPackage("hpddm"):
            raise RuntimeError("SylvesterEKSM: fusing iterative inner solves"
                               " needs PETSc with HPDDM")
        hpddm_type, variant = _PSEUDO_BLOCK_KSP[ksp_type]
        values = {"ksp_type": PETSc.KSP.Type.HPDDM,
                  "ksp_hpddm_type": hpddm_type}
        if variant is not None:
            values["ksp_hpddm_variant"] = variant
        with _temporary_options(inner.getOptionsPrefix(), values):
            inner.setFromOptions()

    def setUp(self, ksp):
        op = ksp.getOperators()[0].getPythonContext()
        self._op = op
        self._dm = op.dm
        self._appctxs = {"K": op.Kctx, "M": op.Mctx}
        self._hook_owners = {"K": _HookOwner(), "M": _HookOwner()}
        self.Kksp = self._create_inner_ksp(ksp, op.K, "K")
        self.Mksp = self._create_inner_ksp(ksp, op.M, "M")
        for which, inner in (("K", self.Kksp), ("M", self.Mksp)):
            # first use: only configure (see firedrake.dmhooks.add_hooks)
            with self._inner(which, save=False):
                inner.setFromOptions()
                if self.fuse:
                    self._fuse_inner_ksp(inner)
            with self._inner(which):
                inner.setUp()
        s = op.S.shape[0]
        self._b = [op.K.createVecRight() for _ in range(s)]
        self._w = [op.K.createVecRight() for _ in range(s)]
        self._dense = {}

    # Block operations for fused solves: dense work matrices per block size

    def _dense_work(self, nb):
        work = self._dense.get(nb)
        if work is None:
            K = self._op.K
            (nloc, n), _ = K.getSizes()

            def dense():
                D = PETSc.Mat().createDense(
                    size=((nloc, n), (PETSc.DECIDE, nb)), comm=K.comm)
                D.setUp()
                D.assemble()
                return D

            work = {"in": dense(), "out": dense(), "K": None, "M": None}
            self._dense[nb] = work
        return work

    def _block_Kmult(self, vecs):
        """[M^{-1} K v for v in vecs]"""
        work = self._dense_work(len(vecs))
        _vecs_to_dense(vecs, work["in"])
        work["K"] = self._op.K.matMult(work["in"], result=work["K"])
        self._op.scale_dense(work["K"], self._op.Dinv)
        with self._inner("M"):
            self.Mksp.matSolve(work["K"], work["out"])
        self._op.scale_dense(work["out"], self._op.Dinv)
        return _dense_to_vecs(work["out"])

    def _block_Ksolve(self, vecs):
        """[K^{-1} M v for v in vecs]"""
        work = self._dense_work(len(vecs))
        _vecs_to_dense(vecs, work["in"])
        self._op.scale_dense(work["in"], self._op.D)
        work["M"] = self._op.M.matMult(work["in"], result=work["M"])
        self._op.scale_dense(work["M"], self._op.D)
        with self._inner("K"):
            self.Kksp.matSolve(work["M"], work["out"])
        return _dense_to_vecs(work["out"])

    def _block_Msolve(self, vecs, out):
        """out[i] = M^{-1} vecs[i]"""
        work = self._dense_work(len(vecs))
        _vecs_to_dense(vecs, work["in"])
        self._op.scale_dense(work["in"], self._op.Dinv)
        with self._inner("M"):
            self.Mksp.matSolve(work["in"], work["out"])
        self._op.scale_dense(work["out"], self._op.Dinv)
        _dense_to_vecs(work["out"], out)

    def solve(self, ksp, b, x):
        op = self._op
        K, S = op.K, op.S
        Kksp, Mksp = self.Kksp, self.Mksp
        p = S.shape[0]
        m = ksp.max_it + 1
        rtol_K0 = Kksp.rtol

        ksp.its = 0
        ksp.setConvergedReason(PETSc.KSP.ConvergedReason.ITERATING)

        # B' = M^{-1} B
        _get_stages(b, self._w)
        if self.fuse:
            self._block_Msolve(self._w, self._b)
        else:
            with self._inner("M"):
                for wi, bi in zip(self._w, self._b):
                    op.scale(wi, op.Dinv)
                    Mksp.solve(wi, bi)
                    op.scale(bi, op.Dinv)

        # B' = Q G (Gram-Schmidt), G = U Sigma W^T. The block size nb is
        # the numerical rank: singular values at round-off level are
        # dropped, as each column of the block costs one solve with K and
        # one with M per iteration.
        Q = _OrthonormalBasis()
        G = numpy.zeros((p + 1, p))
        _extend_basis(lambda v, w: v.copy(w), Q, self._b, 0, G)
        U, sv, Wt = numpy.linalg.svd(G[:p])
        nb = int(numpy.count_nonzero(sv > self.rank_tol * sv[0])) if sv[0] > 0 else 0
        self.rank = nb
        if nb == 0:
            x.zeroEntries()
            ksp.norm = 0.0
            ksp.setConvergedReason(PETSc.KSP.ConvergedReason.CONVERGED_ATOL)
            return

        H = numpy.zeros((nb * (2 * m + 1), nb * 2 * m))
        L = numpy.zeros((nb * (2 * m + 1), nb * 2 * m))
        c = numpy.zeros((nb * (2 * m + 1), p))

        V = _OrthonormalBasis()
        self._V = V
        for j in range(nb):
            V.append(Q.combine(U[:, j], K.createVecRight()))
        c[:nb] = sv[:nb, None] * Wt[:nb]
        del Q

        w = K.createVecLeft()

        def Kmult(v, y):
            K.mult(v, w)
            op.scale(w, op.Dinv)
            with self._inner("M"):
                Mksp.solve(w, y)
            op.scale(y, op.Dinv)

        def Ksolve(v, y):
            op.Mmult(v, w)
            with self._inner("K"):
                Kksp.solve(w, y)

        block_Kmult = self._block_Kmult if self.fuse else None
        block_Ksolve = self._block_Ksolve if self.fuse else None

        # K^{-1} M times the start block
        _extend_basis(Ksolve, V, V[-nb:], 0, L, H, block_op=block_Ksolve)

        best = (numpy.inf, None, -1)
        eps = numpy.finfo(float).eps
        for i in range(ksp.max_it):
            # M^{-1} K times the last block. The K^{-1} M block that pairs
            # with it is only built if the solve goes on.
            _extend_basis(Kmult, V, V[-nb:], (2 * i + 1) * nb, H, L,
                          block_op=block_Kmult)

            # Projected equation T Y + Y S^T = c, T = V^T M^{-1} K V
            nvec = (2 * i + 2) * nb
            T = numpy.linalg.solve(L[:nvec, :nvec].T, H[:nvec + nb, :nvec].T).T
            self._y = solve_sylvester(T[:nvec, :nvec], S.T, c[:nvec])

            # residual of the Sylvester equation, relative per column
            r = T[nvec:nvec + nb, :nvec] @ self._y
            cnorm = numpy.linalg.norm(c[:nb], axis=0)
            cnorm[cnorm == 0] = numpy.linalg.norm(c[:nb])
            rnorms = numpy.linalg.norm(r, axis=0) / (cnorm * (1 + 2 * eps))
            rnorm, rnorm_abs = numpy.linalg.norm(rnorms), numpy.linalg.norm(r)
            if self.residual_fn is not None:
                self.buildSolution(ksp, x)
                rnorm, rnorm_abs = self.residual_fn(x)
                if rnorm < best[0]:
                    best = (rnorm, self._y.copy(), i)

            ksp.monitor(i, rnorm)
            self._convergence_test(ksp, i, rnorm, rnorm_abs)
            if (ksp.getConvergedReason() == 0 and self.residual_fn is not None
                    and i - best[2] >= self.stagnation_its):
                # no progress: the residual is at its attainable level
                ksp.setConvergedReason(PETSc.KSP.ConvergedReason.CONVERGED_ITS)
            if ksp.getConvergedReason() != 0:
                break

            # K^{-1} M times the last block, for the next iteration
            _extend_basis(Ksolve, V, V[-nb:], (2 * i + 2) * nb, L, H,
                          block_op=block_Ksolve)

            if self.adaptive_tol:
                # inexact Krylov: late basis vectors need less accuracy
                Kksp.setTolerances(rtol=max(
                    rtol_K0, min(rtol_K0 / numpy.max(rnorms), self.max_K_rtol)))

        if self.adaptive_tol:
            Kksp.setTolerances(rtol=rtol_K0)
        if best[1] is not None:
            # return the iterate with the smallest residual
            self._y = best[1]
            ksp.norm = best[0]
        self.buildSolution(ksp, x)

    def buildSolution(self, ksp, x):
        for i, xi in enumerate(self._w):
            self._V.combine(self._y[:, i], xi)
        _set_stages(self._w, x)

    def _convergence_test(self, ksp, it, rnorm, rnorm_abs):
        ksp.norm = rnorm
        ksp.its += 1
        ksp.logConvergenceHistory(rnorm)
        if rnorm_abs < ksp.atol:
            ksp.setConvergedReason(PETSc.KSP.ConvergedReason.CONVERGED_ATOL)
        elif rnorm < ksp.rtol:
            ksp.setConvergedReason(PETSc.KSP.ConvergedReason.CONVERGED_RTOL)
        elif rnorm > ksp.divtol:
            ksp.setConvergedReason(PETSc.KSP.ConvergedReason.DIVERGED_DTOL)
        elif it >= ksp.max_it - 1:
            ksp.setConvergedReason(PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT)

    def view(self, ksp, viewer=None):
        if viewer is None or viewer.type != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII("Extended Krylov subspace method for the Sylvester"
                           " equation K X + M X S^T = B\n")
        if self.rank is not None:
            viewer.printfASCII(f"Block size of the last solve: {self.rank}\n")
        if self.fuse:
            viewer.printfASCII("Solves with K and M applied to blocks\n")
        for name, inner in (("K", self.Kksp), ("M", self.Mksp)):
            viewer.printfASCII(f"KSP for the solves with {name}:\n")
            viewer.pushASCIITab()
            inner.view(viewer)
            viewer.popASCIITab()


class IRKSylvesterPC(PCBase):
    """Preconditioner for the stage system of a fully implicit Runge-Kutta
    method that solves a Sylvester equation.

    The stage Jacobian is I x M + dt A x J, with one Jacobian J per stage.
    The preconditioner uses one Jacobian Jbar for all the stages. After
    premultiplication by (dt A)^{-1} x I, it is the Sylvester equation
    K X + M X S^T = B with X the n x s matrix of the stage unknowns,
    S = (dt A)^{-1} - sigma I and K = Jbar + sigma M, where sigma is the
    smallest real part of the eigenvalues of (dt A)^{-1}. K is a backward
    Euler matrix with time step 1/sigma. The equation is solved with
    :class:`SylvesterEKSM`, which needs only solves with K and with M. For a
    linear problem with constant coefficients Jbar = J, and the
    preconditioner is a solver.

    For a nonlinear problem, Jbar is the Jacobian at a combination of the
    stage values (option ``linearisation``), and it is computed again at
    each Newton iteration.

    If some fields have no time derivative (a DAE, e.g. the pressure of
    the incompressible Navier-Stokes equations), M is singular. The
    preconditioner then changes the algebraic unknowns and the constraint
    equations with I + beta S, which is exact, and M is replaced by the
    invertible matrix M_beta = M + beta (the blocks of K that couple to the
    algebraic fields). This needs the algebraic-algebraic block of K to be
    zero. M_beta = D M_1 D with D = diag(I, beta I), so the solver is set up
    for M_1, which does not depend on beta or on the time step, and beta
    enters through diagonal scalings.

    Before the solve, the right-hand side (the n x s matrix R) is replaced
    by its truncated singular value decomposition. The singular values
    below 0.1 times the tolerance of the Sylvester solve, relative to the
    largest one, are dropped. The solve then works with blocks of the
    remaining rank.

    The time step, the shift and S are taken from the stepper at each
    update, so the preconditioner follows changes of the time step.

    Options (prefix ``sylvester_``):

    - ``linearisation`` (``mean``, ``last`` or ``start``): where Jbar is
      evaluated: at the mean of the stage values (default), at the last
      stage value, or at the start of the time step.
    - ``residual`` (``outer`` or ``sylvester``): the residual for the
      convergence test of the Sylvester solve. ``outer`` (default) is the
      residual of the stage system, so that ``sylvester_ksp_rtol`` compares
      with the tolerance of the outer Krylov method. ``sylvester`` is the
      residual of the Sylvester equation, which costs nothing.
    - ``constraint_scale`` (default 1e-2): beta = constraint_scale
      ||M||_F / ||K||_F for DAEs.
    - ``fuse`` (default false): apply the solves with K and M to blocks of
      vectors (iterative inner solvers need PETSc with HPDDM).
    - ``adaptive_tol`` (default true): increase the tolerance of iterative K
      solves as the Sylvester residual decreases.
    - ``ksp_*``: the Sylvester solve. The default is EKSM with
      ``ksp_max_it`` 100 and ``ksp_rtol`` 1e-8.
    - ``K_*``, ``K_mat_type``: the solves with K, configured like any
      Firedrake solver (e.g. geometric multigrid). Default: sparse direct.
    - ``M_*``, ``M_mat_type``: the solves with M (or M_1). Default:
      sparse direct.

    The inner solvers share the application context of the stepper with
    the additional entries ``sylvester_shift`` (sigma) and
    ``sylvester_state`` (the Function at which Jbar is evaluated), for use
    by preconditioners of the inner solves.
    """

    _prefix = "sylvester_"

    # tolerance of the truncation of R, relative to the Sylvester tolerance
    rank_tol_factor = 0.1

    def initialize(self, pc):
        from firedrake import Constant, Function, as_vector, replace
        from firedrake.formmanipulation import split_form
        from ufl.algorithms import expand_derivatives
        from .ufl.deriv import expand_time_derivatives
        from .ufl.manipulation import (remove_time_derivatives,
                                       split_time_derivative_terms)
        from .stage_derivative import StageDerivativeTimeStepper
        from .stage_value import StageValueTimeStepper

        _, P = pc.getOperators()
        octx = get_appctx(pc.getDM())
        stepper = octx.appctx["stepper"]
        if not isinstance(stepper, (StageDerivativeTimeStepper,
                                    StageValueTimeStepper)):
            raise TypeError("IRKSylvesterPC needs a stage-derivative or"
                            " stage-value time stepper")
        self.stepper = stepper
        prefix = (pc.getOptionsPrefix() or "") + self._prefix
        opts = PETSc.Options(prefix)

        u0 = stepper.u0
        V = u0.function_space()
        bcs = stepper.orig_bcs
        self.num_stages = stepper.butcher_tableau.num_stages

        # M: the time-derivative part, K: the rest
        split = split_time_derivative_terms(stepper.F, stepper.t,
                                            timedep_coeffs=(u0,))
        Mform = derivative(remove_time_derivatives(split.time), u0)
        # time derivatives of given functions of t (e.g. a forcing term)
        remainder = expand_time_derivatives(split.remainder, t=stepper.t,
                                            timedep_coeffs=(u0,))
        Jform = derivative(remainder, u0)

        self.linearisation = opts.getString("linearisation", "mean")
        if self.linearisation not in ("start", "last", "mean"):
            raise ValueError(f"-{prefix}linearisation must be start, last"
                             " or mean")
        self.state = u0
        if self.linearisation != "start":
            self.state = Function(V)
            self._update_state()
            Jform = replace(Jform, {u0: self.state})

        self.sigma = Constant(0.0)
        Kform = Jform + self.sigma * Mform

        # DAE: fields without a time derivative
        # (split_form gives the indices (None, None) for one field)
        has_dt = {idx[0] or 0 for idx, f in split_form(Mform)
                  if idx[0] == idx[1] and not f.empty()}
        self.algebraic_fields = [i for i in range(len(V)) if i not in has_dt]
        self.beta = 0.0
        if self.algebraic_fields:
            alg = self.algebraic_fields
            if any(idx[0] in alg and idx[1] in alg and not f.empty()
                   for idx, f in split_form(expand_derivatives(Jform))):
                raise NotImplementedError(
                    "IRKSylvesterPC needs the algebraic-algebraic block of"
                    " the Jacobian to be zero")
            self.constraint_scale = opts.getReal("constraint_scale", 1e-2)
            field_of = numpy.repeat(numpy.arange(len(V)),
                                    [Vi.value_size for Vi in V])

            def restrict(w, fields):
                return as_vector([w[k] if field_of[k] in fields else 0
                                  for k in range(len(field_of))])

            # the blocks of J that couple to the algebraic fields. They are
            # built from the algebraic test function and from the algebraic
            # trial function, so that UFL drops the other terms.
            test, trial = Jform.arguments()
            diff = [i for i in range(len(V)) if i not in alg]
            coupling = (replace(Jform, {test: restrict(test, alg)})
                        + replace(Jform, {test: restrict(test, diff),
                                          trial: restrict(trial, alg)}))
            # ||M||_F for the choice of beta
            from firedrake import assemble
            self._normM = assemble(Mform, bcs=bcs, mat_type="aij").petscmat.norm(
                PETSc.NormType.FROBENIUS)
            # M_1; M_beta = D M_1 D is applied by scaling
            Mform = Mform + coupling

        appctx = dict(octx.appctx,
                      sylvester_shift=self.sigma,
                      sylvester_state=self.state)
        Kmat_type = opts.getString("K_mat_type", "aij")
        Mmat_type = opts.getString("M_mat_type", "aij")
        self.Kctx = self.new_snes_ctx(pc, Kform, bcs, Kmat_type, fcp=octx.fcp,
                                      options_prefix=prefix + "K_",
                                      appctx=appctx)
        self.Mctx = self.new_snes_ctx(pc, Mform, bcs, Mmat_type, fcp=octx.fcp,
                                      options_prefix=prefix + "M_",
                                      appctx=appctx)
        self.K = self.Kctx._assembler_jac.allocate()
        self._assemble_K = self.Kctx._assembler_jac.assemble
        self.M = self.Mctx._assembler_jac.allocate()
        self._assemble_M = self.Mctx._assembler_jac.assemble
        # M is assembled again only if it can change
        self._M_fixed = self._is_fixed(Mform)

        if self.algebraic_fields:
            rstart = self.K.petscmat.getOwnershipRange()[0]
            self._alg_dofs = numpy.concatenate(
                [V.field_ises[i].getIndices() - rstart
                 for i in self.algebraic_fields])

        self._assemble_M(tensor=self.M)
        self.op = None
        self.dt = None
        self._set_time_step()
        self._assemble_K(tensor=self.K)

        self.op = SylvesterOperator(
            self.K.petscmat, self.M.petscmat, self.S, dm=V.dm, Kctx=self.Kctx,
            Mctx=self.Mctx,
            alg_dofs=self._alg_dofs if self.algebraic_fields else None)
        if self.algebraic_fields:
            self.op.set_beta(self.beta)
        opmat = PETSc.Mat().createPython(P.getSizes(), self.op, comm=pc.comm)
        opmat.setUp()

        eksm = SylvesterEKSM(fuse=opts.getBool("fuse", False),
                             adaptive_tol=opts.getBool("adaptive_tol", True))
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.setOperators(opmat)
        ksp.setOptionsPrefix(prefix)
        ksp.setType(PETSc.KSP.Type.PYTHON)
        ksp.setPythonContext(eksm)
        ksp.getPC().setType(PETSc.PC.Type.NONE)
        ksp.setTolerances(rtol=1e-8, max_it=100)
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setFromOptions()
        self.residual = opts.getString("residual", "outer")
        if self.residual not in ("outer", "sylvester"):
            raise ValueError(f"-{prefix}residual must be outer or sylvester")
        if (self.residual == "outer" and ksp.getType() == PETSc.KSP.Type.PYTHON
                and isinstance(ksp.getPythonContext(), SylvesterEKSM)):
            ksp.getPythonContext().residual_fn = self._outer_residual
        self.ksp = ksp

        self._stage_vecs = [self.K.petscmat.createVecRight()
                            for _ in range(self.num_stages)]
        self._work = None
        self.rank = None

    def _is_fixed(self, form):
        """True if the form has no coefficients and no constants, so that
        it does not change."""
        from ufl.algorithms import expand_derivatives
        form = expand_derivatives(form)
        return not form.coefficients() and not form.constants()

    def _stage_weights(self):
        """(w0, w) such that the linearisation state is
        w0 u0 + sum_j w[j] Z_j, with Z_j the stage unknowns."""
        from .stage_derivative import StageDerivativeTimeStepper
        stepper = self.stepper
        s = self.num_stages
        # stage values U = W0 u0 + W Z
        if isinstance(stepper, StageDerivativeTimeStepper):
            W0 = numpy.ones(s)
            W = float(stepper.dt) * numpy.asarray(stepper.butcher_tableau.A,
                                                  dtype=float)
        else:
            vdm = getattr(stepper, "vandermonde", None)
            if vdm is None:
                W0, W = numpy.zeros(s), numpy.eye(s)
            else:
                vdm = numpy.asarray(vdm, dtype=float)
                W0, W = vdm[1:, 0], vdm[1:, 1:]
        c = numpy.eye(s)[-1] if self.linearisation == "last" else numpy.full(s, 1 / s)
        return float(c @ W0), c @ W

    def _update_state(self):
        w0, w = self._stage_weights()
        u0 = self.stepper.u0
        Z = self.stepper.stages.subfunctions
        nf = len(u0.function_space())
        for i, si in enumerate(self.state.subfunctions):
            data = w0 * u0.subfunctions[i].dat.data_ro
            for j, wj in enumerate(w):
                data = data + wj * Z[nf * j + i].dat.data_ro
            si.dat.data[:] = data

    def _set_time_step(self):
        """Compute S, sigma, beta and the stage transformations for the
        current time step."""
        stepper = self.stepper
        dt = float(stepper.dt)
        if dt == self.dt:
            return False
        self.dt = dt
        A = numpy.asarray(stepper.butcher_tableau.A, dtype=float)
        A1, _ = stepper.splitting(A)
        Ainv = numpy.linalg.inv(dt * A)
        sigma = numpy.min(numpy.linalg.eigvals(Ainv).real)
        self.sigma.assign(sigma)
        self.S = Ainv - sigma * numpy.eye(self.num_stages)
        # the stage system is premultiplied by (dt A1)^{-1} x I
        self.A1inv = numpy.linalg.inv(dt * numpy.asarray(A1, dtype=float))
        if self.algebraic_fields:
            # beta = constraint_scale ||M||_F / ||K||_F; K changes with sigma
            self._assemble_K(tensor=self.K)
            self.beta = (self.constraint_scale * self._normM
                         / self.K.petscmat.norm(PETSc.NormType.FROBENIUS))
            self.C = numpy.eye(self.num_stages) + self.beta * self.S
        if self.op is not None:
            self.op.S = self.S
            if self.algebraic_fields:
                self.op.set_beta(self.beta)
        return True

    def update(self, pc):
        if self.linearisation != "start":
            self._update_state()
        self._set_time_step()
        if not self._M_fixed:
            self._assemble_M(tensor=self.M)
        self._assemble_K(tensor=self.K)

    def _stage_matmul(self, B, x, dofs=None):
        """x <- (B x I) x, on the given local dofs of each stage only."""
        s = self.num_stages
        with x as arr:
            a = arr.reshape(s, -1)
            if dofs is None:
                a[:] = B @ a
            else:
                a[:, dofs] = B @ a[:, dofs]

    def _outer_residual(self, xt):
        """Relative and absolute residual of the stage system for the
        candidate solution xt of the Sylvester equation."""
        x, r = self._work[1], self._work[2]
        xt.copy(x)
        if self.algebraic_fields:
            self._stage_matmul(self.C, x, self._alg_dofs)
        self._J.mult(x, r)
        r.aypx(-1.0, self._rhs)
        rabs = r.norm()
        return rabs / self._rhs_norm, rabs

    def _truncate(self, x, y):
        """y = vec(R_r), with R_r the truncated singular value
        decomposition of R, x = vec(R)."""
        vecs = self._stage_vecs
        s = len(vecs)
        _get_stages(x, vecs)
        Q = _OrthonormalBasis()
        G = numpy.zeros((s + 1, s))
        _extend_basis(lambda v, w: v.copy(w), Q, vecs, 0, G)
        U, sv, Wt = numpy.linalg.svd(G[:s])
        tau = self.rank_tol_factor * self.ksp.rtol
        r = int(numpy.count_nonzero(sv > tau * sv[0])) if sv[0] > 0 else 0
        self.rank = r
        if r in (0, s):
            x.copy(y)
            return
        coef = U[:, :r] @ (sv[:r, None] * Wt[:r])
        for j, v in enumerate(vecs):
            Q.combine(coef[:, j], v)
        _set_stages(vecs, y)

    def apply(self, pc, x, y):
        if self._work is None:
            self._work = [x.duplicate() for _ in range(3)]
        self._J = pc.getOperators()[0]
        self._rhs = x
        self._rhs_norm = x.norm() or 1.0
        b = self._work[0]
        self._truncate(x, b)
        self._stage_matmul(self.A1inv, b)
        if self.algebraic_fields:
            # constraint equations times C^T
            self._stage_matmul(self.C, b, self._alg_dofs)
        self.ksp.solve(b, y)
        if self.algebraic_fields:
            # algebraic unknowns: Z = Zt C^T
            self._stage_matmul(self.C, y, self._alg_dofs)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("IRKSylvesterPC has no transpose")

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        viewer.printfASCII("Sylvester equation K X + M X S^T = B for the"
                           " IRK stages\n")
        viewer.printfASCII(f"Jacobian evaluated at: {self.linearisation}\n")
        viewer.printfASCII(f"Shift sigma: {float(self.sigma)}\n")
        if self.algebraic_fields:
            viewer.printfASCII(f"Algebraic fields {self.algebraic_fields},"
                               f" beta: {self.beta}\n")
        if self.rank is not None:
            viewer.printfASCII(f"Rank of the last right-hand side: {self.rank}\n")
        viewer.printfASCII("KSP for the Sylvester equation:\n")
        viewer.pushASCIITab()
        self.ksp.view(viewer)
        viewer.popASCIITab()
