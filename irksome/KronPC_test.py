# suggested setting OMP_NUM_THREADS=1 to improve performance
import os
os.environ["OMP_NUM_THREADS"] = "1"

# -----------------------------
# Imports
# -----------------------------
from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake import *
from firedrake.__future__ import interpolate
from irksome import MeshConstant, Dt, RadauIIA, TimeStepper
from ufl import sin, cos, pi, as_vector, exp, div, dot
import numpy as np


__all__ = ("KronPC",)

# -----------------------------
# KronPC definition
# -----------------------------
class KronPC(PCBase):
    r"""
    Preconditioner applying: y = coef * (A^{-p} \otimes K^{-1}) x
      - A is the Butcher matrix (RadauIIA(s)), p >= 0 (p=0 => I)
      - K is assembled on the single-stage space via kron_operator \in {mass, stiffness}
      - K^{-1} is approximated by a PETSc PC with prefix 'kron_sub_*'

    Options (with this PC's prefix, e.g. -fieldsplit_1_kron_*):
      -kron_coef <real>               : scalar coefficient (default 1.0)
      -kron_pow <int>                 : exponent p for A^{-p} (default 1; p=0 gives identity)
      -kron_operator <mass|stiffness> : which K to assemble (default mass)
      -kron_mat_type <petsc-mat-type> : aux matrix type for K (default aij)
      # sub-PC for K^{-1}:
      -kron_sub_pc_type <...>
      ... (any other -kron_sub_* options forwarded to the sub PC)
    """

    # We rely on the Jacobian P being a python Mat (to get the Firedrake context)
    needs_python_pmat = True

    # --- single-stage form K ---
    def form(self, trial, test, operator_kind):
        if operator_kind == "mass":
            a = inner(trial, test) * dx
        elif operator_kind == "stiffness":
            # tiny mass term for regularization / maybe helpful for AMG solvers  
            a = inner(grad(trial), grad(test)) * dx + 1e-12 * inner(trial, test) * dx
        else:
            raise ValueError(f"Unknown kron_operator '{operator_kind}' (use 'mass' or 'stiffness').")
        bcs = None
        return a, bcs
    # --- ButcherTableau implementation (not generalized yet, to be fixed) ---
    def stage_mat(self, s, pow_):
        if pow_ == 0:
            return np.eye(s)  # A^0 = I
        A = RadauIIA(s).A
        Ainv = np.linalg.inv(A)
        if pow_ == 1:
            return Ainv
        return np.linalg.matrix_power(Ainv, pow_)

    @property
    def num_stages(self):
        return self._s

    # --- PETSc PC interface ---
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type 'python' for KronPC.")

        # Prefix for options
        self._prefix = (pc.getOptionsPrefix() or "") + "kron_"
        opts = PETSc.Options()

        self.coef = opts.getReal(self._prefix + "coef", 1.0)
        pow_ = opts.getInt(self._prefix + "pow", 1)
        if pow_ < 0:
            raise ValueError("kron_pow must be a nonnegative integer.")
        self._pow = pow_

        operator_kind = opts.getString(self._prefix + "operator", "mass")
        mat_type = opts.getString(self._prefix + "mat_type", "aij")

        # Firedrake context & staged space
        _, P = pc.getOperators()
        context = P.getPythonContext()
        Vbig = context.a.arguments()[0].function_space()

        # Work containers
        self.work_in  = Cofunction(Vbig.dual(), name="kron_work_in")  # incoming algebraic vec
        self.work_mid = Function(Vbig,          name="kron_work_mid") # per-stage K^{-1} x
        self.work_out = Function(Vbig,          name="kron_work_out") # after stage mixing

        # Single-stage space (assume uniform across stages ? Need fix !!)
        Vstage = Vbig.sub(0)
        trial = TrialFunction(Vstage)
        test  = TestFunction(Vstage)
        a, bcs = self.form(trial, test, operator_kind)

        # Assemble K
        fc_params = getattr(context, "fc_params", None)
        K = assemble(a, bcs=bcs, mat_type=mat_type, form_compiler_parameters=fc_params)
        self.K = K  # keep reference

        # --- Sub-PC for K^{-1} (by default using LU for now) ---
        sub_pc = PETSc.PC().create(comm=pc.comm)
        sub_pc.incrementTabLevel(1, parent=pc)
        sub_pc.setOptionsPrefix(self._prefix + "sub_")
        sub_pc.setOperators(K.M.handle)   # use K for both A and P
        sub_pc.setFromOptions()           # allow overrides
        # Force LU unless the user overrides via options:
        if sub_pc.getType() is None:
            sub_pc.setType("lu")
        sub_pc.setUp()
        self.sub_pc = sub_pc

        # Stages and dense coupling
        self._s = Vbig.num_sub_spaces()
        self.L = self.stage_mat(self._s, self._pow)

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        r"""
        y = coef * (A^{-p} \otimes K^{-1}) x
        """
        s = self.num_stages

        # 1) Copy PETSc Vec x -> Cofunction(Vbig.dual())
        with self.work_in.dat.vec_wo as vin:
            x.copy(vin)

        # 2) Per-stage action: mid[i] = (sub_pc) in[i]
        for i in range(s):
            with self.work_in.subfunctions[i].dat.vec_ro as xin_i, \
                 self.work_mid.subfunctions[i].dat.vec_wo as mid_i:
                mid_i.set(0.0)
                self.sub_pc.apply(xin_i, mid_i)

        # 3) Dense stage coupling: out[j] = coef * sum_i L[j,i] * mid[i]
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

        # 4) Copy back to PETSc Vec y
        with self.work_out.dat.vec_ro as vout:
            vout.copy(y)

    def applyTranspose(self, pc, x, y):
        # Leave me for now!!
        self.apply(pc, x, y)

    def view(self, pc, viewer=None):
        if viewer is None:
            viewer = PETSc.Viewer.STDOUT(pc.comm)
        viewer.printfASCII("KronPC:\n")
        viewer.printfASCII(f"  stages        : {self.num_stages}\n")
        viewer.printfASCII(f"  coef          : {self.coef}\n")
        viewer.printfASCII(f"  power (p)     : {self._pow}  [A^{-p}]\n")
        viewer.printfASCII("  sub-PC for K^{-1} (kron_sub_*) options:\n")
        if hasattr(self, "sub_pc") and self.sub_pc is not None:
            self.sub_pc.view(viewer)

    def destroy(self, pc):
        if hasattr(self, "sub_pc") and self.sub_pc is not None:
            self.sub_pc.destroy()
            self.sub_pc = None


# -----------------------------
# Problem setup and solve
# -----------------------------
if __name__ == "__main__":
    # Problem parameters
    mesh_size = 4                  # size of mesh
    nu = 1.0 / 50.0                # Viscosity
    T = 2.0                        # Final time
    s = 2                          # Radau IIA stages
    dt_value = T / 10.0            # Time step
    gamma = Constant(50.0)         # Augmentation parameter for testing

    # Mesh and spaces
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size, quadrilateral=True)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q

    # Time variables
    t = Constant(0.0)
    MC = MeshConstant(mesh)
    dt = MC.Constant(dt_value)

    # Manufactured exact solution
    x = SpatialCoordinate(mesh)
    u_exact = 0.5 * exp(T - t) * as_vector([
        sin(pi * x[0]) * cos(pi * x[1]),
        -cos(pi * x[0]) * sin(pi * x[1])
    ])
    p_exact = Constant(0.0)

    # Forcing term: Forcing term f = u_t - \nu \Delta u - u\cdot \nabla u (grad(p) = 0 because p is constant)
    u_t = Dt(u_exact)
    Delta_u = div(grad(u_exact))
    conv_u = dot(grad(u_exact), u_exact)
    f_expr = u_t - nu * Delta_u - conv_u

    # Initial condition
    w = Function(W)
    (u, p) = split(w)
    (u0, p0) = w.subfunctions
    u0.interpolate(u_exact)
    p0.assign(0.0)

    # Boundary conditions
    bc = DirichletBC(W.sub(0), u_exact, "on_boundary")

    # Variational form (with augmented Lagrangian div-div term)
    v, q = TestFunctions(W)
    F = (inner(Dt(u), v) * dx
         + nu * inner(grad(u), grad(v)) * dx
         + inner(dot(grad(u), u), v) * dx
         - inner(p, div(v)) * dx
         + inner(div(u), q) * dx
         - inner(f_expr, v) * dx
         + gamma * inner(div(u), div(v)) * dx) # Augmented Lagrangian Approach

    # Pressure nullspace (constant pressure)
    nsp = [(1, VectorSpaceBasis(constant=True))]

    # Group velocity/pressure stage indices
    velocity_fields = ",".join(str(2*i) for i in range(s))
    pressure_fields = ",".join(str(2*i+1) for i in range(s))

    # Solver parameters
    parameters = {
        # make the global operator mat-free so P is a python Mat
        "mat_type": "matfree",
        "pmat_type": "matfree",

        # outer solve
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-8,
        "ksp_converged_reason": None,

        # ---- Top-level: Schur upper ----
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_factorization_type": "upper",
        "pc_fieldsplit_schur_precondition": "selfp",

        # Map fields
        "pc_fieldsplit_0_fields": velocity_fields,   # (1,1) velocity stages
        "pc_fieldsplit_1_fields": pressure_fields,   # (2,2) pressure Schur

        # ---- (1,1) velocity block ----
        "fieldsplit_0": {
            "ksp_type": "preonly",
            "ksp_rtol": 1e-10,
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_ksp_type":"preonly",
            "assembled_pc_type": "lu",
        },

        # ---- (2,2) Schur block ----
        "fieldsplit_1": {
            "ksp_type": "preonly",
            "ksp_rtol": 1e-10,
            "ksp_converged_reason": None,
            "pc_type": "python",
            "pc_python_type": "__main__.KronPC",   # adjust if KronPC is imported
            # KronPC options:
            "kron_operator": "mass",               # K = mass matrix on single-stage space
            "kron_pow": 0,                         # A^0 = I  => I \otimes (sub_pc) (MassInv-like)
            # nested sub-PC on K:
            "kron_sub_pc_type": "lu",
        },
    }

    # Time stepper
    stepper = TimeStepper(
        F,
        RadauIIA(s),
        t,
        dt,
        w,
        bcs=bc,
        stage_type="deriv",
        solver_parameters=parameters,
        nullspace=nsp
    )

    # Advance in time
    while float(t) < T - 1e-10:
        stepper.advance()
        t.assign(float(t) + float(dt))

    # Report stats
    steps, nonlinear_its, linear_its = stepper.solver_stats()
    print("Total number of timesteps was %d" % (steps))
    print("Average number of nonlinear iterations per timestep was %.2f" % (nonlinear_its/steps))
    print("Average number of linear iterations per timestep was %.2f" % (linear_its/steps))
