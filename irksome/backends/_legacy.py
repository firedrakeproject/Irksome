"""Preserve subfunction operations for backends without tuple selections."""

import ufl


def split_stages(V, stages):
    """Reconstruct symbolic stage expressions with the value shape of ``V``."""
    from irksome.tools import reshape

    if isinstance(stages, (tuple, list)):
        return tuple(stages)
    if len(V) == 1:
        return stages.subfunctions
    stages_np = reshape(stages, (-1, *V.value_shape))
    return [ufl.as_tensor(stage) for stage in stages_np]


def mixed_function(V, functions, Function):
    """Copy the functions into a function on the mixed space ``V``."""
    mixed = Function(V)
    parts = [part for f in functions for part in f.subfunctions]
    for target, part in zip(mixed.subfunctions, parts, strict=True):
        target.assign(part)
    return mixed


def stage_sum(u, stages, weights, alpha=0, indices=None):
    """Return the targets and weighted stage sums for selected subfunctions."""
    num_fields = len(u.subfunctions)
    for i, ubit in enumerate(u.subfunctions):
        if indices is None or i in indices:
            if isinstance(stages, (tuple, list)):
                ks = tuple(k.subfunctions[i] for k in stages)
            else:
                ks = stages.subfunctions[i::num_fields]
            yield ubit, ubit * alpha + sum(k * w for w, k in zip(weights, ks, strict=True))


def assign_stage_sum(u, stages, weights, alpha=0, indices=None):
    """Assign a weighted stage sum plus ``alpha * u`` to selected subfunctions."""
    if alpha == 1:
        for target, expr in stage_sum(u, stages, weights, 0, indices):
            target += expr
    else:
        for target, expr in stage_sum(u, stages, weights, alpha, indices):
            target.assign(expr)


def assign_stages(stages, u, weights, indices=None):
    """Assign weighted copies of selected subfunctions to the stages."""
    num_fields = len(u.subfunctions)
    for s, w in enumerate(weights):
        for i, ubit in enumerate(u.subfunctions):
            if indices is None or i in indices:
                k = stages.subfunctions[num_fields * s + i]
                if abs(float(w)) < 1.e-12:
                    k.zero()
                else:
                    k.assign(ubit * w)
