from ufl import as_ufl

from .base_time_stepper import BaseTimeStepper
from .constant import MeshConstant
from .tableaux.mri_gark_tableaux import MRIGARK
from .tools import extract_timedep_arguments, replace
from .ufl.deriv import expand_time_derivatives
from .ufl.manipulation import split_time_derivative_terms


class MRIGARKTimeStepper(BaseTimeStepper):
    """Front-end class for advancing a time-dependent PDE via an explicit
    multirate infinitesimal GARK method.

    The form is split as ``F = Ffast + Fslow``, where ``Fslow`` carries the
    terms labelled ``explicit`` (or is passed as ``Fexp``) and is treated
    explicitly on the macro step ``dt``.  The remaining terms, including the
    time derivative, are integrated over each fast interval by an inner
    :func:`TimeStepper` with ``num_substeps`` steps.

    The slow terms are explicit, so these methods lose order when the fast
    part is stiff.  The Jacobian of the fast problem is built from its form,
    so ``J`` and ``Jp`` are not accepted.  :meth:`solver_stats` returns the
    number of macro steps and the nonlinear and linear iteration counts of
    the inner stepper; ``num_fast_steps`` counts the fast steps.

    :arg Ffast: A :class:`ufl.Form` with the time derivative and fast terms.
    :arg Fslow: A :class:`ufl.Form` with the slow terms.
    :arg method: A :class:`MRIGARK` instance.
    :arg t: a :class:`Constant` or :class:`Function` on the Real space
        referring to the current time.  It is not modified by ``advance``.
    :arg dt: a :class:`Constant` or :class:`Function` on the Real space
        referring to the macro step size.
    :arg u0: A :class:`Function` containing the current state.
    :arg bcs: An iterable of :class:`DirichletBC`, applied on every fast step.
    :arg fast_parameters: A :class:`dict` with keys ``"tableau"`` (the inner
        :class:`ButcherTableau`, required), ``"num_substeps"`` (fast steps per
        fast interval, default 1) and ``"stepper_kwargs"`` (passed to the
        inner :func:`TimeStepper`).
    """
    def __init__(self, Ffast, Fslow, method, t, dt, u0, bcs=None,
                 fast_parameters=None, appctx=None, nullspace=None,
                 backend="firedrake", **kwargs):
        assert isinstance(method, MRIGARK)
        super().__init__(Ffast + Fslow, t, dt, u0, bcs=bcs, appctx=appctx,
                         nullspace=nullspace, backend=backend)
        if kwargs.get("J") is not None or kwargs.get("Jp") is not None:
            raise NotImplementedError("MRI-GARK does not accept J or Jp")
        if fast_parameters is None or "tableau" not in fast_parameters:
            raise ValueError("MRI-GARK needs fast_parameters with an inner 'tableau'")
        v, u = extract_timedep_arguments(Ffast + Fslow, u0)
        if split_time_derivative_terms(Ffast, timedep_coeffs=(u,)).time.empty():
            raise ValueError("MRI-GARK needs the time derivative in the fast part")
        if not split_time_derivative_terms(Fslow, timedep_coeffs=(u,)).time.empty():
            raise ValueError("MRI-GARK needs a slow part without time derivatives")
        self.method = method
        self.num_substeps = fast_parameters.get("num_substeps", 1)
        assert isinstance(self.num_substeps, int) and self.num_substeps > 0

        s = method.num_stages
        MC = MeshConstant(self.V.mesh(), backend=backend)
        self.t_fast = MC.Constant(0.0)
        self.dt_fast = MC.Constant(0.0)
        self.t_stage = MC.Constant(0.0)
        self.dt_stage = MC.Constant(1.0)
        self.t_slow = [MC.Constant(0.0) for _ in range(s)]
        self.weights = [[MC.Constant(0.0) for _ in range(s)] for _ in method.Gamma]
        self.u_slow = [self._backend.Function(self.V) for _ in range(s)]

        Fslow = expand_time_derivatives(Fslow, t=t, timedep_coeffs=(u,))
        tau = (self.t_fast - self.t_stage) / self.dt_stage
        forcing = sum(
            tau**k * self.weights[k][j] * replace(Fslow, {u: self.u_slow[j], t: self.t_slow[j]})
            for j in range(s) for k in range(len(method.Gamma)))
        F_inner = replace(Ffast, {t: self.t_fast}) + forcing

        bcs_inner = []
        for bc in self.orig_bcs:
            if not hasattr(bc, "_original_arg"):
                raise NotImplementedError("MRI-GARK only supports DirichletBC")
            g = bc._original_arg
            if g == 0:
                bcs_inner.append(bc)
            else:
                bcs_inner.append(bc.reconstruct(g=replace(as_ufl(g), {t: self.t_fast})))

        stepper_kwargs = dict(fast_parameters.get("stepper_kwargs", {}))
        for key, val in kwargs.items():
            stepper_kwargs.setdefault(key, val)

        from .stepper import TimeStepper
        self.fast_stepper = TimeStepper(
            F_inner, fast_parameters["tableau"], self.t_fast, self.dt_fast, u0,
            bcs=bcs_inner, appctx=self.appctx, nullspace=nullspace, **stepper_kwargs)

        self.num_steps = 0
        self.num_fast_steps = 0

    def advance(self):
        method = self.method
        t0 = float(self.t)
        dt = float(self.dt)
        for j in range(method.num_stages):
            self.u_slow[j].assign(self.u0)
            self.t_slow[j].assign(t0 + method.c[j] * dt)
        for i in range(method.num_stages):
            self.u_slow[i].assign(self.u0)
            for k, G in enumerate(method.Gamma):
                for j in range(method.num_stages):
                    self.weights[k][j].assign(G[i, j] / method.dc[i])

            H = method.dc[i] * dt
            self.t_stage.assign(t0 + method.c[i] * dt)
            self.dt_stage.assign(H)
            self.dt_fast.assign(H / self.num_substeps)
            for m in range(self.num_substeps):
                self.t_fast.assign(t0 + method.c[i] * dt + m * H / self.num_substeps)
                self.fast_stepper.advance()
            self.num_fast_steps += self.num_substeps
        self.num_steps += 1

    def solver_stats(self):
        _, nonlinear_its, linear_its = self.fast_stepper.solver_stats()
        return self.num_steps, nonlinear_its, linear_its
