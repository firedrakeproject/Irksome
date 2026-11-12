import copy
import numpy

from .labeling import as_form
from .nystrom_stepper import StageDerivativeNystromTimeStepper
from .tableaux.ButcherTableaux import ButcherTableau, CollocationButcherTableau
from .scheme import GalerkinCollocationScheme, DiscontinuousGalerkinCollocationScheme
from .discontinuous_galerkin_stepper import getElement
from .galerkin_stepper import getTestElement

from firedrake import AuxiliaryOperatorPC, Function, NonlinearVariationalProblem, PETSc, derivative
from firedrake.dmhooks import get_appctx, pop_appctx, push_appctx

try:
    from firedrake import FieldsplitSNES
except ImportError:
    class FieldsplitSNES:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "StageSplitSNES requires firedrake.FieldsplitSNES,"
                " which this version of Firedrake does not provide; please upgrade."
            )

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


class StageSplitSNES(FieldsplitSNES):
    """Apply a nonlinear solve to each stage of a coupled time step.

    Each split contains all spatial fields of one stage. For two stages
    with velocity and pressure, the splits are ``(0, 1)`` and ``(2, 3)``.
    Set ``snes_fieldsplit_type`` to ``additive`` or ``multiplicative`` and
    configure the stage solvers with the ``fieldsplit_`` prefix.

    Supply an auxiliary semidiscrete residual as ``appctx["Fp"]`` and an
    auxiliary time scheme as ``appctx["Fp_scheme"]`` when constructing the
    time stepper. Both default to the original problem. For example, use
    ``appctx={"Fp_scheme": RanaLDScheme(scheme)}`` for a lower triangular
    approximation to the Butcher matrix. The original stage boundary
    conditions are retained. The forcing computed by ``FieldsplitSNES``
    makes solutions of the original residual fixed points of the iteration.

    Notes
    -----
    This solver requires a mixed stage space, as does ``FieldsplitSNES``.
    """

    def initialize(self, snes: PETSc.SNES) -> None:
        """Build the auxiliary problem and its stage solvers.

        Parameters
        ----------
        snes : petsc4py.PETSc.SNES
            The nonlinear solver for the coupled stages.
        """
        ctx = get_appctx(snes.getDM())
        F, bcs = self.form(snes, ctx._x)
        problem = NonlinearVariationalProblem(F, ctx._x, bcs=bcs, form_compiler_parameters=ctx.fcp)
        auxiliary_ctx = ctx.reconstruct(problem=problem, options_prefix=snes.getOptionsPrefix() or "")
        auxiliary_ctx._nullspace = ctx._nullspace
        auxiliary_ctx._nullspace_T = ctx._nullspace_T
        auxiliary_ctx._near_nullspace = ctx._near_nullspace

        # Build the splits from the auxiliary context without changing the outer problem.
        push_appctx(snes.getDM(), auxiliary_ctx)
        try:
            super().initialize(snes)
        finally:
            pop_appctx(snes.getDM(), auxiliary_ctx)
        self._ctx = ctx

    def form(self, snes: PETSc.SNES, stages: Function) -> tuple:
        """Return the auxiliary stage residual and the original boundary conditions.

        Parameters
        ----------
        snes : petsc4py.PETSc.SNES
            The nonlinear solver whose application context contains the stepper.
        stages : firedrake.Function
            The coupled stage unknowns.

        Returns
        -------
        F : ufl.Form
            The stage residual built with ``Fp`` and ``Fp_scheme``.
        bcs : sequence of firedrake.DirichletBC
            The boundary conditions of the original stage problem.
        """
        appctx = self.get_appctx(snes)
        stepper = appctx["stepper"]
        F, _ = stepper.get_form_and_bcs(stages, F=appctx.get("Fp", stepper.F), tableau=appctx.get("Fp_scheme"), bcs=())
        return F, stepper.bigBCs

    def _get_splits(self, snes_options: PETSc.Options, nfields: int) -> list[list[int]]:
        """Group the spatial fields of each stage into one split."""
        stepper = self._ctx.appctx["stepper"]
        nf = len(stepper.V)
        return [list(range(i * nf, (i + 1) * nf)) for i in range(stepper.num_stages)]


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
