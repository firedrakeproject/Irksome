# -----------------------------
# Imports
# -----------------------------
from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake import *
from irksome import RadauIIA
import numpy as np

__all__ = ("KronPC","MassKronPC", "StiffnessKronPC")

# -----------------------------
# KronPC definition
# -----------------------------
class KronPC(PCBase):
    r"""
    Preconditioner applying: y = coef * (A^{-p} \otimes K^{-1}) x
      - A is the Butcher matrix (RadauIIA(s)), p >= 0 (p=0 => I)
      - K is assembled on the single-stage space by subclasses via "form(trial, test)"
      - K^{-1} is approximated by a PETSc PC with prefix 
    """

    needs_python_pmat = True


    def form(self, trial, test):
        """Return (a, bcs) for the single-stage operator K."""
        raise NotImplementedError("KronPC.form() is abstract. Use MassKronPC or StiffnessKronPC (or a custom subclass).\
                                  Subclass must implement 'form(trial, test)'.")

    def stage_mat(self, s, pow_):
        if pow_ == 0:
            return np.eye(s)
        A = RadauIIA(s).A
        Ainv = np.linalg.inv(A)
        if pow_ == 1:
            return Ainv
        return np.linalg.matrix_power(Ainv, pow_)

    @property
    def num_stages(self):
        return self._s

    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type 'python' for KronPC.")

        self._prefix = (pc.getOptionsPrefix() or "") + "kron_"
        opts = PETSc.Options()

        self.coef = opts.getReal(self._prefix + "coef", 1.0)
        pow_ = opts.getInt(self._prefix + "pow", 1)
        if pow_ < 0:
            raise ValueError("kron_pow must be a nonnegative integer.")
        self._pow = pow_

        mat_type = opts.getString(self._prefix + "mat_type", "aij")

        _, P = pc.getOperators()
        context = P.getPythonContext()
        Vbig = context.a.arguments()[0].function_space()

        self.work_in  = Cofunction(Vbig.dual(), name="kron_work_in")
        self.work_mid = Function(Vbig,          name="kron_work_mid")
        self.work_out = Function(Vbig,          name="kron_work_out")

        Vstage = Vbig.sub(0)
        trial = TrialFunction(Vstage)
        test  = TestFunction(Vstage)
        a, bcs = self.form(trial, test)

        fc_params = getattr(context, "fc_params", None)
        K = assemble(a, bcs=bcs, mat_type=mat_type, form_compiler_parameters=fc_params)
        self.K = K

        sub_pc = PETSc.PC().create(comm=pc.comm)
        # Allow users to set options like kron_sub_pc_type hypre
        sub_pc.setOptionsPrefix(self._prefix + "sub_")
        sub_pc.incrementTabLevel(1, parent=pc)
        sub_pc.setOperators(K.M.handle)
        sub_pc.setFromOptions()
        if sub_pc.getType() is None:
            sub_pc.setType("lu")
        sub_pc.setUp()
        self.sub_pc = sub_pc

        self._s = Vbig.num_sub_spaces()
        self.L = self.stage_mat(self._s, self._pow)

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        s = self.num_stages

        with self.work_in.dat.vec_wo as vin:
            x.copy(vin)

        for i in range(s):
            with self.work_in.subfunctions[i].dat.vec_ro as xin_i, \
                 self.work_mid.subfunctions[i].dat.vec_wo as mid_i:
                mid_i.set(0.0)
                self.sub_pc.apply(xin_i, mid_i)

        for j in range(s):
            self.work_out.subfunctions[j].assign(0.0)

        for j in range(s):
            row = self.L[j, :]
            with self.work_out.subfunctions[j].dat.vec_wo as yj:
                for i in range(s):
                    lij = row[i]
                    if lij == 0.0:
                        continue
                    with self.work_mid.subfunctions[i].dat.vec_ro as ui:
                        yj.axpy(self.coef * lij, ui)

        with self.work_out.dat.vec_ro as vout:
            vout.copy(y)

    def applyTranspose(self, pc, x, y):
        self.apply(pc, x, y)

    def view(self, pc, viewer=None):
        if viewer is None:
            viewer = PETSc.Viewer.STDOUT(pc.comm)
        viewer.printfASCII("KronPC:\n")
        viewer.printfASCII(f"  stages        : {self.num_stages}\n")
        viewer.printfASCII(f"  coef          : {self.coef}\n")
        viewer.printfASCII(f"  power (p)     : {self._pow}  [A^{{-p}}]\n")
        viewer.printfASCII("  sub-PC for K^{-1} options:\n")
        if hasattr(self, "sub_pc") and self.sub_pc is not None:
            self.sub_pc.view(viewer)

    def destroy(self, pc):
        if hasattr(self, "sub_pc") and self.sub_pc is not None:
            self.sub_pc.destroy()
            self.sub_pc = None

class MassKronPC(KronPC):
    """K built from the mass form."""
    def form(self, trial, test):
        a = inner(trial, test) * dx
        bcs = None
        return a, bcs


class StiffnessKronPC(KronPC):
    """K built from the (regularized) stiffness form."""
    def form(self, trial, test):
        a = inner(grad(trial), grad(test)) * dx + 1e-12 * inner(trial, test) * dx
        bcs = None
        return a, bcs