from firedrake.solving import _extract_bcs
from firedrake import (DirichletBC, Function, TestFunction,
                       NonlinearVariationalProblem,
                       NonlinearVariationalSolver,
                       replace, inner, dx)
from ufl import as_ufl


def extract_bcs(bcs):
    """Return an iterable of boundary conditions on the residual form"""
    return tuple(bc.extract_form("F") for bc in _extract_bcs(bcs))


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
    return get_sub(Vbig[field + len(V)*i], comp)


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


class BoundsConstrainedDirichletBC(DirichletBC):
    """A DirichletBC with bounds-constrained data."""
    def __init__(self, V, g, sub_domain, bounds, solver_parameters=None):
        if solver_parameters is None:
            solver_parameters = {
                "snes_type": "vinewtonrsls",
                "snes_max_it": 300,
                "snes_atol": 1.e-8,
                "ksp_type": "preonly",
                "mat_type": "aij",
            }
        self.g = g
        self.solver_parameters = solver_parameters
        self.bounds = bounds

        self.gnew = Function(V)
        F = inner(self.gnew - g, TestFunction(V)) * dx
        problem = NonlinearVariationalProblem(F, self.gnew)
        self.solver = NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters)
        super().__init__(V, g, sub_domain)

    @property
    def function_arg(self):
        '''The value of this boundary condition.'''
        self.solver.solve(bounds=self.bounds)
        return self.gnew

    @function_arg.setter
    def function_arg(self, g):
        '''Set the value of this boundary condition.'''
        self.solver.solve(bounds=self.bounds)
        return self.gnew

    def reconstruct(self, V=None, g=None, sub_domain=None):
        V = V or self.function_space()
        g = g or self.g
        sub_domain = sub_domain or self.sub_domain
        return type(self)(V, g, sub_domain, self.bounds, self.solver_parameters)
