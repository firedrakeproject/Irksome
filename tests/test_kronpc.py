import numpy as np
import pytest
import ufl

from functools import reduce
from operator import mul

import irksome  # noqa: F401

from firedrake import (
    UnitSquareMesh, FunctionSpace, TrialFunction, TestFunction, Cofunction,
    Function, as_tensor, inner, grad, dx, solve, errornorm
)


def getA(ns: int):
    """Nonsymmetric, invertible stage matrix."""
    A = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            A[i, j] = 1.0 + 0.1 * (i + 1) - 0.2 * (j + 1)
        A[i, i] += ns
    return A


def random_rhs(Vbig, seed: int = 1234):
    """Random RHS in the dual of Vbig."""
    rng = np.random.default_rng(seed)
    L = Cofunction(Vbig.dual(), name="rhs")

    with L.dat.vec_wo as v:
        arr = rng.standard_normal(v.getSize())
        v.setValues(range(v.getSize()), arr)
        v.assemble()

    return L


def mass_form(A, uu, vv, shift):
    """Stage-coupled mass form corresponding to A kron M."""
    return inner(ufl.dot(A, uu), vv) * dx


def stiffness_form(A, uu, vv, shift):
    """Stage-coupled shifted stiffness form corresponding to A kron K."""
    Au = ufl.dot(A, uu)
    return (
        inner(grad(Au), grad(vv)) * dx
        + shift * inner(Au, vv) * dx
    )


@pytest.mark.parametrize("ns", [2, 3])
@pytest.mark.parametrize(
    ("pc_python_type", "form_builder", "shift", "tol"),
    [
        ("irksome.MassKronPC", mass_form, 0.0, 1.0e-10),
        ("irksome.StiffnessKronPC", stiffness_form, 1.0, 1.0e-10),
    ],
)
def test_kron_pc(ns, pc_python_type, form_builder, shift, tol):
    """
    Solve two ways:
      (1) direct LU on the full mixed operator
      (2) preonly with Python KronPC and inner stage LU
    """
    msh = UnitSquareMesh(2, 2)
    V = FunctionSpace(msh, "CG", 1)

    Vbig = reduce(mul, [V] * ns)

    A_np = getA(ns)
    A = as_tensor(A_np)
    uu = TrialFunction(Vbig)
    vv = TestFunction(Vbig)
    a = form_builder(A, uu, vv, shift)

    L = random_rhs(Vbig, seed=2025)

    u_ref = Function(Vbig, name="u_ref")
    solve(
        a == L,
        u_ref,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    solver_parameters = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": pc_python_type,
        "mat_type": "matfree",
        "kron_sub_pc_type": "lu",
    }

    if pc_python_type == "irksome.StiffnessKronPC":
        solver_parameters["kron_stiffness_shift"] = shift

    u_pc = Function(Vbig, name="u_pc")
    solve(
        a == L,
        u_pc,
        solver_parameters=solver_parameters,
        appctx={"A": A_np},
    )

    err = errornorm(u_ref, u_pc, norm_type="l2")
    assert err < tol, (
        f"{pc_python_type} mismatch: ||u_ref - u_pc|| = {err:.3e} "
        f"(ns={ns})"
    )
