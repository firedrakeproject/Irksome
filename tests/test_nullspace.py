import numpy
import pytest

from firedrake import (
    DirichletBC,
    Function,
    FunctionSpace,
    MixedVectorSpaceBasis,
    TestFunctions,
    TrialFunctions,
    UnitSquareMesh,
    VectorFunctionSpace,
    VectorSpaceBasis,
    assemble,
    div,
    dx,
    grad,
    inner,
)

from irksome.backends.firedrake import getNullspace, get_stage_space


@pytest.mark.parametrize("constant", (True, False))
def test_stokes_stage_nullspace(constant):
    mesh = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    Z = V * Q

    Vbig = get_stage_space(Z, 2)
    u0, p0, u1, p1 = TrialFunctions(Vbig)
    v0, q0, v1, q1 = TestFunctions(Vbig)
    a = sum((
        inner(grad(u), grad(v)) * dx
        - inner(p, div(v)) * dx
        - inner(div(u), q) * dx
        for u, p, v, q in ((u0, p0, v0, q0), (u1, p1, v1, q1))
    ))

    zero = Function(V).assign(0)
    bcs = [
        DirichletBC(Vbig.sub(0), zero, "on_boundary"),
        DirichletBC(Vbig.sub(2), zero, "on_boundary"),
    ]
    A = assemble(a, bcs=bcs)

    if constant:
        pressure_basis = VectorSpaceBasis(constant=True, comm=mesh.comm)
        source_norm = None
    else:
        source = Function(Q).assign(1)
        with source.dat.vec_ro as source_vec:
            source_norm = source_vec.norm()
        pressure_basis = VectorSpaceBasis([source])

    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), pressure_basis])
    stage_nullspace = getNullspace(Z, Vbig, 2, nullspace)

    if source_norm is not None:
        with source.dat.vec_ro as source_vec:
            assert numpy.isclose(source_vec.norm(), source_norm)

    pressure_bases = [stage_nullspace._bases[i] for i in (1, 3)]
    assert all(not basis._constant for basis in pressure_bases)
    assert all(len(basis._vecs) == 1 for basis in pressure_bases)
    assert all(basis.is_orthonormal() for basis in pressure_bases)

    stage_nullspace._apply(A)
    petsc_nullspace = A.petscmat.getNullSpace()
    assert len(petsc_nullspace.getVecs()) == 2
    assert petsc_nullspace.test(A.petscmat)
