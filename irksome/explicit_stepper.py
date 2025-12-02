from .dirk_stepper import DIRKTimeStepper


# We can reuse the DIRK stepper to do one-stage at a time, but since we're
# just solving a mass matrix at each time step we can optimize to
# never rebuild the jacobian or preconditioner.
class ExplicitTimeStepper(DIRKTimeStepper):
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None,
                 **kwargs):
        assert butcher_tableau.is_explicit
        # we just have one mass matrix we're reusing for each time step and
        # each stage, so we can nudge this along
        if solver_parameters is None:
            solver_parameters = {}
        solver_parameters["snes_lag_jacobian_persists"] = "true"
        solver_parameters["snes_lag_jacobian"] = -2
        solver_parameters["snes_lag_preconditioner_persists"] = "true"
        solver_parameters["snes_lag_preconditioner"] = -2
        super().__init__(F, butcher_tableau, t, dt, u0, bcs=bcs,
                         solver_parameters=solver_parameters, **kwargs)
