from functools import partial
from firedrake import (DirichletBC, Function, TestFunction,
                       NonlinearVariationalProblem,
                       NonlinearVariationalSolver,
                       replace, inner, dx)
from ufl import as_ufl


def get_sub(u, indices):
    for i in indices:
        if i is not None:
            u = u.sub(i)
    return u


def bc2space(bc, V):
    return get_sub(V, bc._indices)


def stage2spaces4bc(bc, V, Vbig, i):
    """used to figure out how to apply Dirichlet BC to each stage"""
    sub = 0 if len(V) == 1 else bc.function_space_index()
    comp = (bc.function_space().component,)
    Vbigi = get_sub(Vbig[sub + len(V)*i], comp)
    return Vbigi


def BCStageData(V, gcur, u0, u0_mult, i, t, dt):
    if gcur == 0:
        # special case DirichletBC(V, 0, ...), do nothing
        return gcur
    if V.component is None:
        indices = (V.index,)
    else:
        indices = (V.parent.index, V.component)
    gdat = gcur - u0_mult[i] * get_sub(u0, indices)
    return gdat


def EmbeddedBCData(bc, t, dt, num_fields, butcher_tableau, ws, u0):
    gorig = bc._original_arg
    if gorig == 0:
        # special case DirichletBC(V, 0, ...), do nothing
        return gorig

    V = u0.function_space()
    sub = 0 if len(V) == 1 else bc.function_space_index()
    comp = (bc.function_space().component,)
    btilde = butcher_tableau.btilde

    g = replace(as_ufl(gorig), {t: t + dt}) - bc2space(bc, u0)
    g -= sum(get_sub(ws[sub + len(V)*j], comp) * (btilde[j] * dt)
             for j in range(butcher_tableau.num_stages))
    return g


class BoundsConstrainedBC(DirichletBC):
    """A DirichletBC with bounds-constrained data."""
    def __init__(self, V, g, sub_domain, bounds, solver_parameters=None):
        super().__init__(V, g, sub_domain)
        if solver_parameters is None:
            solver_parameters = {
                "snes_type": "vinewtonssls",
            }
        self.solver_parameters = solver_parameters
        self.bounds = bounds

    @property
    def function_arg(self):
        '''The value of this boundary condition.'''
        if hasattr(self, "_function_arg_update"):
            self._function_arg_update()
        return self._function_arg

    @function_arg.setter
    def function_arg(self, g):
        '''Set the value of this boundary condition.'''
        V = self.function_space()
        gnew = Function(V)
        try:
            # Use the interpolant as initial guess
            gnew.interpolate(g)
        except (NotImplementedError, AttributeError):
            pass
        F = inner(TestFunction(V), gnew - g) * dx
        problem = NonlinearVariationalProblem(F, gnew)
        solver = NonlinearVariationalSolver(problem,
                                            solver_parameters=self.solver_parameters)

        self._function_arg = gnew
        self.function_arg_update = partial(solver.solve, bounds=self.bounds)
        self.function_arg_update()
