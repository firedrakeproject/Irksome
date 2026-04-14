from abc import abstractmethod
from firedrake import (
    derivative, replace, lhs, rhs, Function, TrialFunction,
    LinearVariationalProblem, LinearVariationalSolver,
    NonlinearVariationalProblem, NonlinearVariationalSolver,
)
from firedrake.petsc import PETSc
from .labeling import as_form, as_linear_form
from .tools import AI, getNullspace, flatten_dats
from .backend import get_backend
import ufl


class BaseTimeStepper:
    """Base class for various time steppers.  This is mainly to give code reuse stashing
    objects that are common to all the time steppers.  It's a developer-level class.
    """
    def __init__(self, F, t, dt, u0,
                 bcs=None, appctx=None, nullspace=None, backend: str = "firedrake"):
        self._backend = get_backend(backend)
        self.F = F
        self.t = t
        self.dt = dt
        self.u0 = u0
        if bcs is None:
            bcs = ()
        self.orig_bcs = bcs
        self.nullspace = nullspace
        self.V = self._backend.get_function_space(u0)

        appctx_base = {"stepper": self}

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
            :class:firedrake.TestFunction`. To specify a linear problem,
            `F` must be of the form a(t; w, v) - L(t; v) where
            `w` is a :class:firedrake.TrialFunction`.
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg num_stages: The number of stages to solve for.  It could be the number of
            RK stages or relate to the polynomial degree (Galerkin)
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` or `firedrake.EquationBC`
            containing the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the RK method.  Support for `firedrake.EquationBC` is limited
            to the stage derivative formulation with DAE style BCs.
    :arg Fp: A :class:`ufl.Form` instance to precondition the semi-discrete linearization.
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
                 bcs=None, Fp=None, solver_parameters=None,
                 appctx=None, nullspace=None,
                 transpose_nullspace=None, near_nullspace=None,
                 splitting=None, bc_type=None,
                 butcher_tableau=None, bounds=None,
                 **kwargs):

        use_linear_variational_solver = False
        if len(as_form(F).arguments()) == 2:
            F = as_linear_form(F, u0)
            use_linear_variational_solver = True

        super().__init__(F, t, dt, u0,
                         bcs=bcs, appctx=appctx, nullspace=nullspace)

        self.num_stages = num_stages
        if butcher_tableau:
            assert num_stages == butcher_tableau.num_stages
        if splitting is None:
            splitting = AI
        self.splitting = splitting
        self.bc_type = bc_type

        self.num_steps = 0
        self.num_nonlinear_iterations = 0
        self.num_linear_iterations = 0

        stages = self.get_stages()
        self.stages = stages

        Fbig, bigBCs = self.get_form_and_bcs(stages)
        Jpbig = None
        if Fp is not None:
            Fp = as_linear_form(Fp, u0)
            Fpbig, _ = self.get_form_and_bcs(stages, F=Fp, bcs=())
            Jpbig = derivative(Fpbig, stages)

        V = u0.function_space()
        Vbig = stages.function_space()
        nullspace = getNullspace(V, Vbig, num_stages, nullspace)
        transpose_nullspace = getNullspace(V, Vbig, num_stages, transpose_nullspace)
        near_nullspace = getNullspace(V, Vbig, num_stages, near_nullspace)

        self.bigBCs = bigBCs

        if use_linear_variational_solver:
            Fbig = replace(Fbig, {stages: TrialFunction(stages.function_space())})
            abig = lhs(Fbig)
            Lbig = rhs(Fbig)
            problem = LinearVariationalProblem(
                abig, Lbig, stages, bcs=bigBCs, aP=Jpbig,
                form_compiler_parameters=kwargs.pop("form_compiler_parameters", None),
                constant_jacobian=kwargs.pop("constant_jacobian", False),
                restrict=kwargs.pop("restrict", False),
            )
            solver_constructor = LinearVariationalSolver
            # FIXME
            solver_constructor = NonlinearVariationalSolver
        else:
            problem = NonlinearVariationalProblem(
                Fbig, stages, bcs=bigBCs, Jp=Jpbig,
                form_compiler_parameters=kwargs.pop("form_compiler_parameters", None),
                is_linear=kwargs.pop("is_linear", False),
                restrict=kwargs.pop("restrict", False),
            )
            solver_constructor = NonlinearVariationalSolver
        self.problem = problem
        self.solver = solver_constructor(
            self.problem, appctx=self.appctx,
            nullspace=nullspace,
            transpose_nullspace=transpose_nullspace,
            near_nullspace=near_nullspace,
            solver_parameters=solver_parameters,
            **kwargs,
        )

        # stash these for later in case we do bounds constraints
        self.stage_bounds = self.get_stage_bounds(bounds)

    def advance(self):
        """Advances the system from time `t` to time `t + dt`.
        Note: overwrites the value `u0`."""

        self.solver.solve(bounds=self.stage_bounds)

        self.num_steps += 1
        self.num_nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.num_linear_iterations += self.solver.snes.getLinearSolveIterations()
        self._update()

    # allow butcher tableau as input for preconditioners to create
    # an alternate operator
    @abstractmethod
    def get_form_and_bcs(self, stages, tableau=None, F=None):
        pass

    def solver_stats(self):
        return (self.num_steps, self.num_nonlinear_iterations, self.num_linear_iterations)

    def get_stages(self) -> ufl.Coefficient:
        return self._backend.get_stages(self.V, self.num_stages)

    def get_stage_bounds(self, bounds=None):
        if bounds is None:
            return None

        Vbig = self.stages.function_space()
        bounds_type, lower, upper = bounds
        if lower is None:
            slb = Function(Vbig).assign(PETSc.NINFINITY)
        if upper is None:
            sub = Function(Vbig).assign(PETSc.INFINITY)

        if bounds_type == "stage":
            if lower is not None:
                dats = [lower.dat] * (self.num_stages)
                slb = Function(Vbig, val=flatten_dats(dats))
            if upper is not None:
                dats = [upper.dat] * (self.num_stages)
                sub = Function(Vbig, val=flatten_dats(dats))

        elif bounds_type == "last_stage":
            V = self.u0.function_space()
            if lower is not None:
                ninfty = Function(V).assign(PETSc.NINFINITY)
                dats = [ninfty.dat] * (self.num_stages-1)
                dats.append(lower.dat)
                slb = Function(Vbig, val=flatten_dats(dats))
            if upper is not None:
                infty = Function(V).assign(PETSc.INFINITY)
                dats = [infty.dat] * (self.num_stages-1)
                dats.append(upper.dat)
                sub = Function(Vbig, val=flatten_dats(dats))

        else:
            raise ValueError("Unknown bounds type")

        return (slb, sub)
