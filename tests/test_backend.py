import types

import numpy as np
import pytest
from firedrake import (
    Function, FunctionSpace, TensorFunctionSpace, UnitSquareMesh,
    VectorFunctionSpace,
)

from irksome.backend import get_backend
from irksome.tools import assign_stage_sum, assign_stages, mixed_function, split_stages


@pytest.fixture
def backend(tuple_subfunctions):
    return get_backend("firedrake")


@pytest.fixture(scope="module", params=["scalar", "vector", "tensor", "mixed"])
def space(request):
    mesh = UnitSquareMesh(1, 1)
    if request.param == "scalar":
        return FunctionSpace(mesh, "CG", 1)
    if request.param == "vector":
        return VectorFunctionSpace(mesh, "CG", 1)
    if request.param == "tensor":
        return TensorFunctionSpace(mesh, "CG", 1)
    return VectorFunctionSpace(mesh, "CG", 2) * FunctionSpace(mesh, "DG", 0)


def test_backend_selection():
    module = types.ModuleType("custom_backend")
    assert get_backend(module) is module
    assert get_backend("firedrake").__name__ == "irksome.backends.firedrake"


def test_custom_backend_stage_operations(space):
    backend = types.ModuleType("custom_backend")
    fd_backend = get_backend("firedrake")
    backend.split_stages = fd_backend.split_stages
    backend.assign_stages = fd_backend.assign_stages
    backend.assign_stage_sum = fd_backend.assign_stage_sum
    backend.mixed_function = fd_backend.mixed_function
    u = Function(space).assign(2)
    stages = Function(space * space)
    assign_stages(stages, u, [1, -1], backend=backend)
    assert len(split_stages(space, stages, backend=backend)) == 2
    assign_stage_sum(u, stages, [2, 3], alpha=1, backend=backend)
    for part in u.subfunctions:
        np.testing.assert_allclose(part.dat.data_ro, 0)
    mixed = mixed_function(space * space, [u, u], backend=backend)
    for part in mixed.subfunctions:
        np.testing.assert_allclose(part.dat.data_ro, 0)


@pytest.mark.parametrize("num_stages", [1, 3])
def test_stage_operations(backend, space, num_stages):
    u = Function(space)
    for i, part in enumerate(u.subfunctions):
        part.assign(i + 1)
    stages = backend.get_stages(space, num_stages)
    weights = np.arange(num_stages) - 1
    backend.assign_stages(stages, u, weights)
    for s in range(num_stages):
        for i in range(len(space)):
            part = stages.subfunctions[len(space) * s + i]
            np.testing.assert_allclose(part.dat.data_ro, (i + 1) * weights[s])

    ks = split_stages(space, stages, backend=backend)
    assert len(ks) == num_stages
    assert all(k.ufl_shape == u.ufl_shape for k in ks)
    if isinstance(ks[-1], Function):
        assert all(k is other for k, other in zip(ks, split_stages(space, stages, backend=backend)))
        ks[-1].assign(u)
        weights[-1] = 1

    backend.assign_stage_sum(u, stages, np.arange(num_stages) + 1, alpha=0.5)
    scale = 0.5 + sum(weights * (np.arange(num_stages) + 1))
    for i, part in enumerate(u.subfunctions):
        np.testing.assert_allclose(part.dat.data_ro, (i + 1) * scale)


@pytest.mark.parametrize("indices", [(0,), (1,), (1, 0), ()])
def test_selected_stage_operations(backend, indices):
    mesh = UnitSquareMesh(1, 1)
    V = VectorFunctionSpace(mesh, "CG", 2) * FunctionSpace(mesh, "DG", 0)
    u = Function(V).assign(2)
    stages = backend.get_stages(V, 2).assign(3)
    backend.assign_stages(stages, u, [0, -2], indices=indices)
    backend.assign_stage_sum(u, stages, [2, -3], alpha=0.5, indices=indices)
    for i, part in enumerate(u.subfunctions):
        np.testing.assert_allclose(part.dat.data_ro, 13 if i in indices else 2)
    for s in range(2):
        for i in range(len(V)):
            expected = [0, -4][s] if i in indices else 3
            np.testing.assert_allclose(stages.subfunctions[len(V) * s + i].dat.data_ro, expected)


def test_stage_function_sum(backend, space):
    u = Function(space).assign(4)
    stages = [Function(space).assign(1), Function(space).assign(2)]
    backend.assign_stage_sum(u, stages, [0, 0], alpha=1)
    for part in u.subfunctions:
        np.testing.assert_allclose(part.dat.data_ro, 4)
    backend.assign_stage_sum(u, stages, [2, -3], alpha=1)
    for part in u.subfunctions:
        np.testing.assert_allclose(part.dat.data_ro, 0)


def test_mixed_function(backend, space):
    functions = [Function(space).assign(i + 1) for i in range(3)]
    mixed = mixed_function(backend.get_stage_space(space, 3), (f for f in functions), backend=backend)
    for s, f in enumerate(functions):
        for i, part in enumerate(f.subfunctions):
            target = mixed.subfunctions[len(space) * s + i]
            np.testing.assert_array_equal(target.dat.data_ro, part.dat.data_ro)
    functions[0].assign(7)
    np.testing.assert_allclose(mixed.subfunctions[0].dat.data_ro, 1)
