
import numpy
from ufl import as_ufl, as_tensor, Form, Coefficient
from .tableaux import ButcherTableaux
from .constant import vecconst
from .tools import AI, dot, replace, reshape, fields_to_components
from .ufl.deriv import Dt, TimeDerivative, expand_time_derivatives
from .backend import get_backend

__all__ = ["getForm"]

def getForm(F: Form, butch:ButcherTableaux, t: Coefficient, dt:Coefficient, u0:Coefficient, stages, bcs=None, bc_type=None, splitting=AI, aux_indices=None, backend:str="firedrake"):
    """Given a time-dependent variational form and a
    :class:`ButcherTableau`, produce UFL for the s-stage RK method.

    :arg F: a :class:`ufl.Form` instance describing the semi-discrete problem.
    :arg butch: the :class:`ButcherTableau` for the RK method being used to
         advance in time.
    :arg t: a :class:`Function` or :class:`Constant` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time.
    :arg dt: a :class:`Function` or :class:`Constant` on the Real space over the same mesh as
         `u0`.  This serves as a variable referring to the current time step.
         The user may adjust this value between time steps.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg stages: a :class:`Function` representing the stages to be solved for.
    :kwarg bcs: optionally, a :class:`DirichletBC` or :class:`EquationBC`
            object (or iterable thereof) containing (possibly time-dependent)
            boundary conditions imposed on the system.
        :kwarg bc_type: How to manipulate the strongly-enforced boundary
            conditions to derive the stage boundary conditions.  Should
            be a string, either "DAE", which implements BCs as
            constraints in the style of a differential-algebraic
            equation, or "ODE", which takes the time derivative of the
            boundary data and evaluates this for the stage values.
            Support for `firedrake.EquationBC` in `bcs` is limited
            to DAE style BCs.
        :kwarg splitting: a callable that maps the (floating point) Butcher matrix
            to a pair of matrices `A1, A2` such that `butch.A = A1 A2`.  This is used
            to vary between the classical RK formulation and Butcher's reformulation
            that leads to a denser mass matrix with block-diagonal stiffness.
            Some choices of function will assume that `butch.A` is invertible.
        :kwarg aux_indices: a list of field indices to be discretized as :class:`TimeDerivative`,
            analogous to :class:`ContinouosPetrovGalerkinTimeStepper`.

    :returns: a 2-tuple of
       - `Fnew`, the :class:`Form`
       - `bcnew`, a list of :class:`firedrake.DirichletBC` or :class:`EquationBC`
         objects to be posed on the stages
    """
    backend_cls = get_backend(backend)
    if bc_type is None:
        bc_type = "DAE"

    # preprocess time derivatives
    F = expand_time_derivatives(F, t=t, timedep_coeffs=(u0,))
    v, = F.arguments()
    V = backend_cls.get_function_space(v)
    assert V ==  backend_cls.get_function_space(u0)

    c = vecconst(butch.c, backend=backend)
    bA1, bA2 = splitting(butch.A)
    try:
        bA2inv = numpy.linalg.inv(bA2)
    except numpy.linalg.LinAlgError:
        raise NotImplementedError("We require A = A1 A2 with A2 invertible")
    A1 = vecconst(bA1, backend=backend)
    A2inv = vecconst(bA2inv, backend=backend)

    # s-way product space for the stage variables
    num_stages = butch.num_stages
    Vbig = backend_cls.get_function_space(stages)
    test = backend_cls.TestFunction(Vbig)

    # set up the pieces we need to work with to do our substitutions
    v_np = reshape(test, (num_stages, *v.ufl_shape))
    w_np = reshape(stages, (num_stages, *u0.ufl_shape))
    A1w = dot(A1, w_np)
    A2invw = dot(A2inv, w_np)
    dtu = TimeDerivative(u0)

    aux_components = fields_to_components(V, aux_indices or [])

    repl = {}
    for i in range(num_stages):
        usub = u0 + as_tensor(A1w[i]) * dt
        dtusub = A2invw[i]
        if aux_components:
            # Apply TimeDerivative substitution to auxiliary fields
            usub = reshape(usub, u0.ufl_shape)
            usub[aux_components] = dtusub[aux_components] * dt

        repl[i] = {t: t + c[i] * dt,
                   v: v_np[i],
                   u0: usub,
                   dtu: dtusub}

    Fnew = sum(replace(F, repl[i]) for i in range(num_stages))

    if bcs is None:
        bcs = []
    if bc_type == "ODE":
        assert splitting == AI, "ODE-type BC aren't implemented for this splitting strategy"

        def bc2stagebc(bc, i):
            from irksome.bcs import BCStageData
            from firedrake.bcs import EquationBCSplit

            if isinstance(bc, EquationBCSplit):
                raise NotImplementedError("EquationBC not implemented for ODE formulation")
            gorig = as_ufl(bc._original_arg)
            gfoo = expand_time_derivatives(Dt(gorig), t=t, timedep_coeffs=(u0,))
            gcur = replace(gfoo, {t: t + c[i] * dt})
            return BCStageData(bc, gcur, u0, stages, i)

    elif bc_type == "DAE":
        try:
            bA1inv = numpy.linalg.inv(bA1)
            A1inv = vecconst(bA1inv, backend=backend)
        except numpy.linalg.LinAlgError:
            raise NotImplementedError("Cannot have DAE BCs for this Butcher Tableau/splitting")

        def bc2stagebc(bc, i):
            from irksome.bcs import BCStageData, stage2spaces4bc, bc2space
            from firedrake.bcs import EquationBCSplit, EquationBC
            if isinstance(bc, EquationBCSplit):
                F_bc_orig = expand_time_derivatives(bc.f, t=t, timedep_coeffs=(u0,))
                F_bc_new = replace(F_bc_orig, repl[i])
                Vbigi = stage2spaces4bc(bc, V, Vbig, i)
                return EquationBC(F_bc_new == 0, stages, bc.sub_domain, V=Vbigi,
                                  bcs=[bc2stagebc(innerbc, i) for innerbc in backend_cls.extract_bcs(bc.bcs)])
            else:
                gcur = bc._original_arg
                if gcur != 0:
                    gorig = as_ufl(gcur)
                    ucur = bc2space(bc, u0)
                    gcur = (1/dt) * sum((replace(gorig, {t: t + c[j]*dt}) - ucur) * A1inv[i, j]
                                        for j in range(num_stages))
                return BCStageData(bc, gcur, u0, stages, i)
    else:
        raise ValueError(f"Unrecognised bc_type: {bc_type}")

    # This logic uses information set up in the previous section to
    # set up the new BCs for either method
    bcs = backend_cls.extract_bcs(bcs)
    bcnew = [bc2stagebc(bc, i) for i in range(num_stages) for bc in bcs]

    return Fnew, bcnew
