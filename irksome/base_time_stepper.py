import numpy

from abc import abstractmethod
from firedrake import Function, NonlinearVariationalProblem, NonlinearVariationalSolver
from firedrake.dmhooks import pop_parent, push_parent
from firedrake.petsc import PETSc
from .tools import AI, get_stage_space, getNullspace
from pyop2.types import MixedDat


class BaseTimeStepper:
    """Base class for various time steppers.  This is mainly to give code reuse stashing
    objects that are common to all the time steppers.  It's a developer-level class.
    """
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


class StageCoupledTimeStepper(BaseTimeStepper):
    """This developer-level class provides common features used by
    various methods requiring stage-coupled variational problems to
    compute the stages (e.g. fully implicit RK, Galerkin-in-time)

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg num_stages: The number of stages to solve for.  It could be the number of
            RK stages or relate to the polynomial degree (Galerkin)
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing
            the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the RK method.
    :arg solver_parameters: An optional :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    :arg appctx: An optional :class:`dict` containing application context.
            This gets included with particular things that Irksome will
            pass into the nonlinear solver so that, say, user-defined preconditioners
            have access to it.
    :arg nullspace: A list of tuples of the form (index, VSB) where
            index is an index into the function space associated with
            `u` and VSB is a :class: `firedrake.VectorSpaceBasis`
            instance to be passed to a
            `firedrake.MixedVectorSpaceBasis` over the larger space
            associated with the Runge-Kutta method
    :arg splitting: An optional kwarg (not used by all superclasses)
    :arg bc_type: An optional kwarg (not used by all superclasses)
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
            the Runge-Kutta method to be used for time marching.
    :arg bounds: An optional kwarg used in certain bounds-constrained methods.
    """
    def __init__(self, F, t, dt, u0, num_stages,
                 bcs=None, solver_parameters=None,
                 appctx=None, nullspace=None,
                 splitting=None, bc_type=None,
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
