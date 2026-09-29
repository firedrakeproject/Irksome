from .backend import get_backend
from ufl import as_ufl, replace


def get_sub(u, indices):
    for i in indices:
        if i is not None:
            u = u.sub(i)
    return u


def bc2space(bc, V):
    return get_sub(V, bc._indices)


def stage2spaces4bc(bc, V, Vbig, i):
    """used to figure out how to apply Dirichlet BC to each stage"""
    field = 0 if len(V) == 1 else bc.function_space_index()
    comp = (bc.function_space().component,)
    return get_sub(Vbig[field + len(V) * i], comp)


def BCStageData(bc, gcur, u0, stages, i):
    if bc._original_arg == 0:
        gcur = 0
    V = u0.function_space()
    Vbig = stages.function_space()
    Vbigi = stage2spaces4bc(bc, V, Vbig, i)
    return bc.reconstruct(V=Vbigi, g=gcur)


def EmbeddedBCData(bc, butcher_tableau, t, dt, u0, stages):
    Vbc = bc2space(bc, u0.function_space())
    gorig = bc._original_arg
    if gorig == 0:
        g = gorig
    else:
        V = u0.function_space()
        field = 0 if len(V) == 1 else bc.function_space_index()
        comp = (bc.function_space().component,)
        ws = stages.subfunctions[field::len(V)]
        btilde = butcher_tableau.btilde
        num_stages = butcher_tableau.num_stages
        g = replace(as_ufl(gorig), {t: t + dt}) - gorig
        g -= sum(get_sub(ws[j], comp) * (btilde[j] * dt) for j in range(num_stages))
    return bc.reconstruct(V=Vbc, g=g)


def BoundsConstrainedDirichletBC(V, g, sub_domain, bounds, solver_parameters=None, backend="firedrake"):
    """A DirichletBC with bounds-constrained data."""
    backend_class = get_backend(backend)
    return backend_class.create_bounds_constrained_bc(V, g, sub_domain, bounds, solver_parameters=solver_parameters)
