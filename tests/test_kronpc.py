import numpy as np
import pytest
from functools import reduce
from operator import mul


from firedrake import (
    UnitSquareMesh, FunctionSpace, TrialFunction, TestFunction, Cofunction,
    Function, as_tensor, inner, dx, solve, errornorm, ufl
)

def getA(ns: int):
    """nonsymmetric, invertible matrix."""
    A = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            A[i, j] = 1.0 + 0.1*(i+1) - 0.2*(j+1)  # breaks symmetry
        A[i, i] += ns 
    return A


def random_rhs(Vbig, seed: int = 1234):
    """random RHS in the dual of Vbig. Yes, we need the dual space here!!"""
    rng = np.random.default_rng(seed)
    L = Cofunction(Vbig.dual(), name="rhs")
    with L.dat.vec_wo as v:
        arr = rng.standard_normal(v.getSize())
        v.setValues(range(v.getSize()), arr)
        v.assemble()
    return L

@pytest.mark.parametrize("ns", [2,3,4,5]) # meaningful cases are ns >= 2
def test_mass_kron_pc(ns):
    """
    Solve two ways:
      (1) direct LU on the full mixed operator
      (2) preonly with Python PC = MassKronPC and inner stage LU
    """
    # Create mesh and function-space
    msh = UnitSquareMesh(2, 2)
    V = FunctionSpace(msh, "CG", 1)

    # Stage-stacked space V
    Vbig = reduce(mul, [V] * ns)

    # Build global operator
    A_np = getA(ns)          
    A = as_tensor(A_np)      
    uu = TrialFunction(Vbig)
    vv = TestFunction(Vbig)
    a = inner(ufl.dot(A, uu), vv) * dx

    # The RHS
    L = random_rhs(Vbig, seed=2025)

    # Reference solution via full LU
    u_ref = Function(Vbig, name="u_ref")
    solve(
        a == L,
        u_ref,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "lu"
        },
    )

    # Exact inverse via our KronPC:
    u_pc = Function(Vbig, name="u_pc")
    solve(
        a == L,
        u_pc,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "irksome.MassKronPC",  # per professor
            "mat_type": "matfree",
            "kron": {
                "pow": 1,
                "coef": 1.0,
                "sub":{
                    "pc_type": "lu"
                    },
        }
        },
        appctx={"A": A_np},
    )

    err = errornorm(u_ref, u_pc, norm_type="l2")
    assert err < 1.0e-10, f"MassKronPC mismatch: ||u_ref - u_pc|| = {err:.3e} (ns={ns})"
