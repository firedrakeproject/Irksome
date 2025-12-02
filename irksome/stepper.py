from .scheme import ContinuousPetrovGalerkinScheme, DiscontinuousGalerkinScheme
from .dirk_stepper import DIRKTimeStepper
from .explicit_stepper import ExplicitTimeStepper
from .discontinuous_galerkin_stepper import DiscontinuousGalerkinTimeStepper
from .galerkin_stepper import ContinuousPetrovGalerkinTimeStepper
from .imex import RadauIIAIMEXMethod, DIRKIMEXMethod
from .labeling import split_explicit
from .stage_derivative import StageDerivativeTimeStepper, AdaptiveTimeStepper
from .stage_value import StageValueTimeStepper
from .tools import AI

valid_base_kwargs = ("bcs", "form_compiler_parameters", "is_linear", "restrict", "solver_parameters",
                     "nullspace", "transpose_nullspace", "near_nullspace",
                     "appctx", "options_prefix", "pre_apply_bcs")

valid_kwargs_per_stage_type = {
    "deriv": ["Fp", "stage_type", "bc_type", "splitting", "adaptive_parameters", "aux_indices"],
    "value": ["Fp", "stage_type", "basis_type",
              "update_solver_parameters", "splitting", "bounds", "use_collocation_update"],
    "dirk": ["Fp", "stage_type"],
    "explicit": ["Fp", "stage_type"],
    "imex": ["Fexp", "stage_type", "it_solver_parameters", "prop_solver_parameters",
             "splitting", "num_its_initial", "num_its_per_step"],
    "dirkimex": ["Fexp", "stage_type", "mass_parameters"],
    "dg": ["Fp"],
    "cpg": ["Fp", "bc_type", "aux_indices"]}

valid_adapt_parameters = ["tol", "dtmin", "dtmax", "KI", "KP",
                          "max_reject", "onscale_factor",
                          "safety_factor", "gamma0_params"]


def imex_separation(F, Fexp_kwarg, label):
    Fimp, Fexp_label = split_explicit(F)
    if Fexp_kwarg is None:
        if Fexp_label is None:
            raise ValueError(f"Calling an {label} scheme with no explicit form.  Did you really mean to do this?")
        else:
            Fexp = Fexp_label
    else:
        Fexp = Fexp_kwarg
        if Fexp_label is not None:
            raise ValueError("You specified an explicit part in two ways!")

    return Fimp, Fexp


def TimeStepper(F, method, t, dt, u0, **kwargs):
    """Helper function to dispatch between various back-end classes
       for doing time stepping.  Returns an instance of the
       appropriate class.

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` iss the
            :class:firedrake.TestFunction`.
    :arg method: A :class:`ButcherTableau` instance (for RK methods) or
            a :class:`GalerkinScheme` instance (for CPG or DG) methods
            to be used in time marching.
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
    :kwarg aux_indices: Only valid for continuous Petrov Galerkin time scheme.  It
            specifies that some of the variables in `u0` are to be treated as
            auxiliary, that is, discretized in the lower-order DG test space.
    """
    # first pluck out the cases for Galerkin in time...

    if isinstance(method, DiscontinuousGalerkinScheme):
        assert set(kwargs.keys()).issubset(list(valid_base_kwargs) + valid_kwargs_per_stage_type["dg"])
        return DiscontinuousGalerkinTimeStepper(F, method, t, dt, u0, **kwargs)
    elif isinstance(method, ContinuousPetrovGalerkinScheme):
        assert set(kwargs.keys()).issubset(list(valid_base_kwargs) + valid_kwargs_per_stage_type["cpg"])
        return ContinuousPetrovGalerkinTimeStepper(F, method, t, dt, u0, **kwargs)

    stage_type = kwargs.pop("stage_type", "deriv")
    adapt_params = kwargs.pop("adaptive_parameters", None)
    if adapt_params is not None:
        assert stage_type == "deriv", "Adaptive time stepping is only implemented for derivative stage type"

    base_kwargs = {}
    for k in valid_base_kwargs:
        if k in kwargs:
            base_kwargs[k] = kwargs.pop(k)
    bcs = base_kwargs.pop("bcs", None)

    for cur_kwarg in kwargs.keys():
        if cur_kwarg not in valid_kwargs_per_stage_type[stage_type]:
            raise ValueError(f"kwarg {cur_kwarg} is not allowable for stage_type {stage_type}")

    if stage_type == "deriv":
        Fp = kwargs.get("Fp", None)
        bc_type = kwargs.get("bc_type", "DAE")
        splitting = kwargs.get("splitting", AI)
        aux_indices = kwargs.get("aux_indices", None)
        if adapt_params is None:
            return StageDerivativeTimeStepper(
                F, method, t, dt, u0, bcs, Fp=Fp,
                bc_type=bc_type, splitting=splitting, aux_indices=aux_indices, **base_kwargs)
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
            F, method, t, dt, u0, bcs,
            bc_type=bc_type, splitting=splitting,
            tol=tol, dtmin=dtmin, dtmax=dtmax, KI=KI, KP=KP,
            max_reject=max_reject, onscale_factor=onscale_factor,
            safety_factor=safety_factor, gamma0_params=gamma0_params,
            **base_kwargs)
    elif stage_type == "value":
        Fp = kwargs.get("Fp", None)
        splitting = kwargs.get("splitting", AI)
        basis_type = kwargs.get("basis_type")
        update_solver_parameters = kwargs.get("update_solver_parameters")
        bounds = kwargs.get("bounds")
        use_collocation_update = kwargs.get("use_collocation_update", False)
        return StageValueTimeStepper(
            F, method, t, dt, u0, bcs=bcs, Fp=Fp,
            splitting=splitting, basis_type=basis_type,
            update_solver_parameters=update_solver_parameters,
            bounds=bounds, use_collocation_update=use_collocation_update,
            **base_kwargs)
    elif stage_type == "dirk":
        Fp = kwargs.get("Fp", None)
        return DIRKTimeStepper(
            F, method, t, dt, u0, bcs, Fp=Fp, **base_kwargs)
    elif stage_type == "explicit":
        Fp = kwargs.get("Fp", None)
        return ExplicitTimeStepper(
            F, method, t, dt, u0, bcs, Fp=Fp, **base_kwargs)
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
            Fimp, Fexp, method, t, dt, u0, bcs,
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
            Fimp, Fexp, method, t, dt, u0, bcs,
            solver_parameters, mass_parameters, appctx, nullspace)
