from .backend import get_backend
import ufl
from irksome.tools import get_sub


def BCStageData(bc, gcur, u0, stages, i, backend="firedrake"):
    backend_cls = get_backend(backend)
    if bc._original_arg == 0:
        gcur = 0
    V = backend_cls.get_function_space(u0)
    Vbig = backend_cls.get_function_space(stages)
    Vbigi = backend_cls.stage2spaces4bc(bc, V, Vbig, i)
    return bc.reconstruct(V=Vbigi, g=gcur)


def EmbeddedBCData(bc, butcher_tableau, t, dt, u0, stages, backend="firedrake"):
    backend_cls = get_backend(backend)
    Vbc = backend_cls.bc2space(bc, backend_cls.get_function_space(u0))
    gorig = bc._original_arg
    if gorig == 0:
        g = gorig
    else:
        V = backend_cls.get_function_space(u0)
        field = 0 if len(V) == 1 else bc.function_space_index()
        comp = (bc.function_space().component,)
        ws = backend_cls.extract_subfunctions(stages)[field::len(V)]
        btilde = butcher_tableau.btilde
        num_stages = butcher_tableau.num_stages
        g = ufl.replace(ufl.as_ufl(gorig), {t: t + dt}) - gorig
        g -= sum(get_sub(ws[j], comp) * (btilde[j] * dt) for j in range(num_stages))
    return bc.reconstruct(V=Vbc, g=g)


def BoundsConstrainedDirichletBC(V, g, sub_domain, bounds, solver_parameters=None, backend="firedrake"):
    """A DirichletBC with bounds-constrained data."""
    backend_class = get_backend(backend)
    return backend_class.create_bounds_constrained_bc(V, g, sub_domain, bounds, solver_parameters=solver_parameters)
