from functools import partial
from firedrake import (DirichletBC, Function, TestFunction,
                       NonlinearVariationalProblem,
                       NonlinearVariationalSolver,
                       replace, inner, dx)


def get_sub(u, indices):
    for i in indices:
        u = u.sub(i)
    return u


def bc2space(bc, V):
    return get_sub(V, bc._indices)


def stage2spaces4bc(bc, V, Vbig, i):
    """used to figure out how to apply Dirichlet BC to each stage"""
    num_fields = len(V)
    sub = 0 if num_fields == 1 else bc.function_space_index()
    comp = bc.function_space().component

    Vbigi = Vbig[sub+num_fields*i]
    if comp is not None:  # check for sub-piece of vector-valued
        Vbigi = Vbigi.sub(comp)

    return Vbigi


def BCStageData(V, gcur, u0, u0_mult, i, t, dt):
    if V.component is None:  # V is not a bit of a VFS
        if V.index is None:  # not part of MFS, either
            indices = ()
        else:  # part of MFS
            indices = (V.index,)
    else:  # bottommost space is bit of VFS
        if V.parent.index is None:  # but not part of a MFS
            indices = (V.component,)
        else:   # V is a bit of a VFS inside an MFS
            indices = (V.parent.index, V.component)

    if gcur == 0:  # special case DirichletBC(V, 0, ...), do nothing
        gdat = gcur
    else:
        gdat = gcur - u0_mult[i] * get_sub(u0, indices)
    return gdat


def EmbeddedBCData(bc, t, dt, num_fields, butcher_tableau, ws, u0):
    gorig = bc._original_arg
    if gorig == 0:  # special case DirichletBC(V, 0, ...), do nothing
        gdat = gorig
    else:
        gcur = replace(gorig, {t: t+dt})
        sub = 0 if num_fields == 1 else bc.function_space_index()
        comp = bc.function_space().component
        num_stages = butcher_tableau.num_stages
        btilde = butcher_tableau.btilde
        if comp is None:  # check for sub-piece of vector-valued
            for j in range(num_stages):
                gcur -= dt*btilde[j]*ws[num_fields*j+sub]
        else:
            for j in range(num_stages):
                gcur -= dt*btilde[j]*ws[num_fields*j+sub].sub(comp)

        gdat = gcur - bc2space(bc, u0)
    return gdat


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
