import numpy

from abc import abstractmethod
from firedrake import Function, NonlinearVariationalProblem, NonlinearVariationalSolver
from firedrake.dmhooks import pop_parent, push_parent
from firedrake.petsc import PETSc
from .tools import AI, get_stage_space, getNullspace
from pyop2.types import MixedDat


class BaseTimeStepper:
    def __init__(self, F, t, dt, u0,
                 bcs=None, appctx=None, nullspace=None):
        self.F = F
        self.t = t
        self.dt = dt
        self.u0 = u0
        if bcs is None:
            bcs = ()
        self.orig_bcs = bcs
        self.nullspace = nullspace
        self.V = u0.function_space()

        appctx_base = {
            "F": F,
            "t": t,
            "dt": dt,
            "u0": u0,
            "bcs": bcs,
            "nullspace": nullspace}

        if appctx is None:
            self.appctx = appctx_base
        else:
            self.appctx = {**appctx, **appctx_base}

    @abstractmethod
    def advance(self):
        pass

    @abstractmethod
    def solver_stats(self):
        pass


# Stage derivative + stage value + (maybe?) RadauIIAIMEX
class StageCoupledTimeStepper(BaseTimeStepper):
    def __init__(self, F, t, dt, u0, num_stages,
                 bcs=None, solver_parameters=None,
                 appctx=None, nullspace=None,
                 splitting=None, bc_type="DAE",
                 butcher_tableau=None, bounds=None):

        super().__init__(F, t, dt, u0,
                         bcs=bcs, appctx=appctx, nullspace=nullspace)

        self.num_stages = num_stages
        if butcher_tableau:
            assert num_stages == butcher_tableau.num_stages
            self.appctx["butcher_tableau"] = butcher_tableau
        if splitting is None:
            splitting = AI
        self.splitting = splitting
        self.appctx["splitting"] = splitting
        self.bc_type = bc_type

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        stages = self.get_stages()
        self.stages = stages

        Fbig, bigBCs = self.get_form_and_bcs(self.stages)

        nsp = getNullspace(u0.function_space(),
                           stages.function_space(),
                           self.num_stages, nullspace)

        self.bigBCs = bigBCs

        self.prob = NonlinearVariationalProblem(Fbig, stages, bigBCs)

        push_parent(self.u0.function_space().dm, self.stages.function_space().dm)
        self.solver = NonlinearVariationalSolver(
            self.prob, appctx=self.appctx, nullspace=nsp,
            solver_parameters=solver_parameters)
        pop_parent(self.u0.function_space().dm, self.stages.function_space().dm)

        # stash these for later in case we do bounds constraints
        self.stage_bounds = self.get_stage_bounds(bounds)

    def advance(self):
        """Advances the system from time `t` to time `t + dt`.
        Note: overwrites the value `u0`."""

        push_parent(self.u0.function_space().dm, self.stages.function_space().dm)
        self.solver.solve(bounds=self.stage_bounds)
        pop_parent(self.u0.function_space().dm, self.stages.function_space().dm)

        self.num_steps += 1
        self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()
        self._update()

    # allow butcher tableau as input for preconditioners to create
    # an alternate operator
    @abstractmethod
    def get_form_and_bcs(self, stages, butcher_tableau=None):
        pass

    def solver_stats(self):
        return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)

    def get_stages(self):
        Vbig = get_stage_space(self.V, self.num_stages)
        return Function(Vbig)

    def get_stage_bounds(self, bounds=None):
        if bounds is None:
            return None

        flatten = numpy.hstack
        Vbig = self.stages.function_space()
        bounds_type, lower, upper = bounds
        if lower is None:
            slb = Function(Vbig).assign(PETSc.NINFINITY)
        if upper is None:
            sub = Function(Vbig).assign(PETSc.INFINITY)

        if bounds_type == "stage":
            if lower is not None:
                dats = [lower.dat] * (self.num_stages)
                slb = Function(Vbig, val=MixedDat(flatten(dats)))
            if upper is not None:
                dats = [upper.dat] * (self.num_stages)
                sub = Function(Vbig, val=MixedDat(flatten(dats)))

        elif bounds_type == "last_stage":
            V = self.u0.function_space()
            if lower is not None:
                ninfty = Function(V).assign(PETSc.NINFINITY)
                dats = [ninfty.dat] * (self.num_stages-1)
                dats.append(lower.dat)
                slb = Function(Vbig, val=MixedDat(flatten(dats)))
            if upper is not None:
                infty = Function(V).assign(PETSc.INFINITY)
                dats = [infty.dat] * (self.num_stages-1)
                dats.append(upper.dat)
                sub = Function(Vbig, val=MixedDat(flatten(dats)))

        else:
            raise ValueError("Unknown bounds type")

        return (slb, sub)
