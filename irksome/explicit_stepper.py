from .dirk_stepper import DIRKTimeStepper


# We can reuse the DIRK stepper to do one-stage at a time, but since we're
# just solving a mass matrix at each time step we can optimize to
# never rebuild the jacobian or preconditioner.
class ExplicitTimeStepper(DIRKTimeStepper):
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None,
                 appctx=None):
        assert butcher_tableau.is_explicit
        # we just have one mass matrix we're reusing for each time step and
        # each stage, so we can nudge this along
        solver_parameters = {} if solver_parameters is None else solver_parameters
        solver_parameters["snes_lag_jacobian_persists"] = "true"
        solver_parameters["snes_lag_jacobian"] = -2
        solver_parameters["snes_lag_preconditioner_persists"] = "true"
        solver_parameters["snes_lag_preconditioner"] = -2
        super(ExplicitTimeStepper, self).__init__(
            F, butcher_tableau, t, dt, u0, bcs=bcs,
            solver_parameters=solver_parameters, appctx=appctx,
            nullspace=None)

    # DAE treatment of the boundary conditions in this case says that
    # we should impose the BCs so that they are satisfied for the next
    # stage for all but that last stage, and that they are satisfied
    # for the next timestep for the last stage.
    def update_bc_constants(self, i, c):
        """This sets the BCs that are imposed on the solution at stage i.  In
           the explicit case, we say that the stage i solution should
           match the prescribed boundary values at time c[i+1], and at
           the next time step for the last stage.  So, we set c to be
           the (i+1)^st stage time, and a_vals and d_vals to
           correspond to the (i+1)^st row of A.  For the last stage,
           we use c = 1 for the next time-step, and use the row b to
           impose the final solution satisfies the BCs there.
        """

        AAb = self.AAb
        CCone = self.CCone
        a_vals, d_val = self.bc_constants
        ns = AAb.shape[1]
        for j in range(i):
            a_vals[j].assign(AAb[i+1, j])
        for j in range(i, ns):
            a_vals[j].assign(0)
        d_val.assign(AAb[i+1, i])
        c.assign(CCone[i+1])
