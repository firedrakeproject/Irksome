from .dirk_stepper import DIRKTimeStepper
from .explicit_stepper import ExplicitTimeStepper
from .imex import RadauIIAIMEXMethod, DIRKIMEXMethod
from .labeling import split_explicit
from .stage_derivative import StageDerivativeTimeStepper, AdaptiveTimeStepper
from .stage_value import StageValueTimeStepper
from .tools import AI

valid_base_kwargs = ("form_compiler_parameters", "is_linear", "restrict", "solver_parameters",
                     "nullspace", "transpose_nullspace", "near_nullspace",
                     "appctx", "options_prefix", "pre_apply_bcs")

valid_kwargs_per_stage_type = {
    "deriv": ["stage_type", "bc_type", "splitting", "adaptive_parameters"],
    "value": ["stage_type", "basis_type",
              "update_solver_parameters", "splitting", "bounds", "use_collocation_update"],
    "dirk": ["stage_type", "bcs", "nullspace", "solver_parameters", "appctx"],
    "explicit": ["stage_type", "bcs", "solver_parameters", "appctx"],
    "imex": ["Fexp", "stage_type", "it_solver_parameters", "prop_solver_parameters",
             "splitting", "num_its_initial", "num_its_per_step"],
    "dirkimex": ["Fexp", "stage_type", "mass_parameters"]}

valid_adapt_parameters = ["tol", "dtmin", "dtmax", "KI", "KP",
                          "max_reject", "onscale_factor",
                          "safety_factor", "gamma0_params"]


def imex_separation(F, Fexp1, label):
    Fimp, Fexp0 = split_explicit(F)
    if Fexp0 is None:
        if Fexp1 is None:
            raise ValueError(f"Calling an {label} scheme with no explicit form.  Did you really mean to do this?")
        else:
            Fexp = Fexp1
    else:
        Fexp = Fexp0
        if Fexp1 is not None:
            raise ValueError("You specified an explicit part in two ways!")

    return Fimp, Fexp


def TimeStepper(F, butcher_tableau, t, dt, u0, **kwargs):
    """Helper function to dispatch between various back-end classes
       for doing time stepping.  Returns an instance of the
       appropriate class.

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` iss the
            :class:firedrake.TestFunction`.
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
            the Runge-Kutta method to be used for time marching.
    :arg t: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` or
            :class: `firedrake.EquationBC` containing
            the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the RK method.
    :arg nullspace: A list of tuples of the form (index, VSB) where
            index is an index into the function space associated with
            `u` and VSB is a :class: `firedrake.VectorSpaceBasis`
            instance to be passed to a
            `firedrake.MixedVectorSpaceBasis` over the larger space
            associated with the Runge-Kutta method
    :arg stage_type: Whether to formulate in terms of a stage
            derivatives or stage values. Support for `firedrake.EquationBC`
            in `bcs` is limited to the stage derivative formulation.
    :arg splitting: An callable used to factor the Butcher matrix
    :arg bc_type: For stage derivative formulation, how to manipulate
            the strongly-enforced boundary conditions.
            Support for `firedrake.EquationBC` in `bcs` is limited
            to DAE style BCs.
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    :arg update_solver_parameters: A :class:`dict` of parameters for
            inverting the mass matrix at each step (only used if
            stage_type is "value")
    :arg adaptive_parameters: A :class:`dict` of parameters for use with
            adaptive time stepping (only used if stage_type is "deriv")
    :arg use_collocation_update: An optional kwarg indicating whether to use
        the terminal value of the collocation polynomial as the solution
        update. This is needed to bypass the mass matrix inversion when
        enforcing bounds constraints with an RK method that is not stiffly
        accurate. Currently, only constant-in-time boundary conditions are
        supported.
    """
    stage_type = kwargs.pop("stage_type", "deriv")
    adapt_params = kwargs.pop("adaptive_parameters", None)
    if adapt_params is not None:
        assert stage_type == "deriv", "Adaptive time stepping is only implemented for derivative stage type"

    base_kwargs = {}
    for k in valid_base_kwargs:
        if k in kwargs:
            base_kwargs[k] = kwargs.pop(k)
    bcs = kwargs.pop("bcs", None)

    for cur_kwarg in kwargs.keys():
        assert cur_kwarg in valid_kwargs_per_stage_type[stage_type]

    if stage_type == "deriv":
        bc_type = kwargs.get("bc_type", "DAE")
        splitting = kwargs.get("splitting", AI)
        if adapt_params is None:
            return StageDerivativeTimeStepper(
                F, butcher_tableau, t, dt, u0, bcs,
                bc_type=bc_type, splitting=splitting, **base_kwargs)
        else:
            for param in adapt_params:
                assert param in valid_adapt_parameters
            tol = adapt_params.get("tol", 1e-3)
            dtmin = adapt_params.get("dtmin", 1.e-15)
            dtmax = adapt_params.get("dtmax", 1.0)
            KI = adapt_params.get("KI", 1/15)
            KP = adapt_params.get("KP", 0.13)
            max_reject = adapt_params.get("max_reject", 10)
            onscale_factor = adapt_params.get("onscale_factor", 1.2)
            safety_factor = adapt_params.get("safety_factor", 0.9)
            gamma0_params = adapt_params.get("gamma0_params")
        return AdaptiveTimeStepper(
            F, butcher_tableau, t, dt, u0, bcs,
            bc_type=bc_type, splitting=splitting,
            tol=tol, dtmin=dtmin, dtmax=dtmax, KI=KI, KP=KP,
            max_reject=max_reject, onscale_factor=onscale_factor,
            safety_factor=safety_factor, gamma0_params=gamma0_params,
            **base_kwargs)
    elif stage_type == "value":
        splitting = kwargs.get("splitting", AI)
        basis_type = kwargs.get("basis_type")
        update_solver_parameters = kwargs.get("update_solver_parameters")
        bounds = kwargs.get("bounds")
        use_collocation_update = kwargs.get("use_collocation_update", False)
        return StageValueTimeStepper(
            F, butcher_tableau, t, dt, u0, bcs=bcs,
            splitting=splitting, basis_type=basis_type,
            update_solver_parameters=update_solver_parameters,
            bounds=bounds, use_collocation_update=use_collocation_update,
            **base_kwargs)
    elif stage_type == "dirk":
        return DIRKTimeStepper(
            F, butcher_tableau, t, dt, u0, bcs, **base_kwargs)
    elif stage_type == "explicit":
        return ExplicitTimeStepper(
            F, butcher_tableau, t, dt, u0, bcs, **base_kwargs)
    elif stage_type == "imex":
        Fimp, Fexp = imex_separation(F, kwargs.get("Fexp"), stage_type)
        appctx = base_kwargs.get("appctx")
        nullspace = base_kwargs.get("nullspace")
        splitting = kwargs.get("splitting", AI)
        it_solver_parameters = kwargs.get("it_solver_parameters")
        prop_solver_parameters = kwargs.get("prop_solver_parameters")
        num_its_initial = kwargs.get("num_its_initial", 0)
        num_its_per_step = kwargs.get("num_its_per_step", 0)

        return RadauIIAIMEXMethod(
            Fimp, Fexp, butcher_tableau, t, dt, u0, bcs,
            it_solver_parameters, prop_solver_parameters,
            splitting, appctx, nullspace,
            num_its_initial, num_its_per_step)
    elif stage_type == "dirkimex":
        Fimp, Fexp = imex_separation(F, kwargs.get("Fexp"), stage_type)
        appctx = base_kwargs.get("appctx")
        nullspace = base_kwargs.get("nullspace")
        solver_parameters = base_kwargs.get("solver_parameters")
        mass_parameters = kwargs.get("mass_parameters")
        return DIRKIMEXMethod(
            Fimp, Fexp, butcher_tableau, t, dt, u0, bcs,
            solver_parameters, mass_parameters, appctx, nullspace)
