from firedrake import *
from irksome.fetsome.timesplitting import split_time_form, split_time_form_on
from irksome.fetsome.fetutils import strip_dt_form
from irksome.fetsome.fetutils import max_time_order, dt_n, spacetime_dot

# Class that generates the initial condition, intermediate and final forms
# for a finite element in time problem
class TimeFormGenerator:
    # Default initialiser that saves fundamental information used by the form generator
    def __init__(self, F, u, v, dt, time_quadrature, T, Tprime, stage_mappings):
        # Number of trial and test stages must be the same
        if len(stage_mappings[u]) != len(stage_mappings[v]):
            raise AssertionError("Trial and test stage mapping dimensions do not coincide, "
                                 "got #trial: " + len(stage_mappings[u]) + " and "
                                 "#test: " + len(stage_mappings[v]))
        self.F = F
        self.trial = u
        self.test = v
        self.dt = dt
        self.quadrature = time_quadrature
        self.T = T
        self.Tprime = Tprime
        self.stage_mappings = stage_mappings


# Base class for a form generator that builds spacetime forms through the test-function
# splitting algorithm.
# Splits form on the test function and substitutes the mixed space expansions for the trial function u
class SplitTimeFormGenerator(TimeFormGenerator):
    def __init__(self, F, u, v, dt, time_quadrature, T, Tprime, stage_mappings):
        # Base initialisation
        super().__init__(F, u, v, dt, time_quadrature, T, Tprime, stage_mappings)

        # Generator specific initialisation
        splitted = split_time_form_on(self.F, self.test)
        if splitted[0]:
            raise AssertionError("Found term not linear in trial function, variational form cannot "
                                 "be composed.")
        self._splitted = splitted[1:]

        max_order = max([max_time_order(form, self.trial) for form in self._splitted])

        if max_order < 0:
            raise AssertionError("No dependence in the trial function found to generate mixed space "
                                 "forms.")

        u_expanded = self._compose_u_expanded(max_order)
        self._u_expanded_form_mat = self._expand_u_in_splitted(u_expanded)
    
    # Helper function to expand u into a linear combination of the timepoint values and
    # time basis functions
    def _compose_u_expanded(self, max_order):
        uhatis = split(self.stage_mappings[self.trial])
        tabs = self.quadrature.tabulate_at_points(self.T, max_order)
        u_expanded = {}
        for n in range(max_order + 1):
            basis_tau = tabs[n]

            # The expanded expression contains the transformation factor from
            # global coordinates to local coordinates d/dt -> d/dtau
            u_expanded[n] = [Constant((1/self.dt)**n) * spacetime_dot(basis_tau[:, q], uhatis)
                             for q in range(self.quadrature.num_points)]

        return u_expanded
    
    # Helper function to expand the test function splitted term for each timepoint
    def _expand_u_in_splitted(self, u_expanded):
        u_expanded_form_mat = [[zero() for _ in range(len(self._splitted))]
                               for _ in range(self.quadrature.num_points)]
        
        for v_order, form in enumerate(self._splitted):
            # Skip the forms that are zero
            if form == zero():
                continue

            # What is the maximum time order of the trial function to be expanded?
            u_order = max_time_order(form, self.trial)
            if u_order < 0:
                    raise AssertionError("No trial function dependence found in a sub-form to be generated.")

            for q in range(self.quadrature.num_points):
                # Prepare all time derivative replacements at the quadrature points
                # and replace them in the form
                repl_dict = {}
                for n in reversed(range(u_order + 1)):
                    dt_expr = dt_n(self.trial, n)
                    u_dt_n = u_expanded[n][q]
                    repl_dict[dt_expr] = u_dt_n
                time_expanded_form = replace(form, repl_dict)

                # Save the expanded form for the quadrature point
                u_expanded_form_mat[q][v_order] = time_expanded_form

        return u_expanded_form_mat
    
    # Helper function to compose all entries of the main form vector
    def _compose_interior_form_block(self, vhatrs):
        # Prepare tabulation of the time test function space and timepoint restricted space functions
        test_basis_tabs = self.quadrature.tabulate_at_points(self.Tprime, len(self._splitted) - 1)

        form_blocks = []
        # Compose the interior forms
        for r, vhatr in enumerate(vhatrs):
            interior_form = zero()

            # Reduce the expanded form matrix in its quadrature and test function order dimensions
            for q in range(self.quadrature.num_points):
                for n, expanded_form in enumerate(self._u_expanded_form_mat[q]):
                    # Skip the zero forms
                    if expanded_form == zero():
                        continue
                    
                    test_basis_tab = test_basis_tabs[n]
                    dt_psi_s = (1/self.dt)**(n - 1) * test_basis_tab[r, q]
                    clean_form = strip_dt_form(expanded_form)
                    interior_form += Constant(self.quadrature.weights[q]) * Constant(dt_psi_s) * replace(clean_form , {self.test: vhatr})

            form_blocks.append(interior_form)

        return form_blocks
    
    # Helper function to compose all entries of the forcing function vector
    def _compose_forcing_form_block(self, vhatrs, fs):
        if fs == []:
            # Homogenous forcing term if no term is supplied
            fs = [Constant(0) * self.test * dx for _ in range(len(split(self.stage_mappings[self.trial])))]
        
        P_M = self.quadrature.time_mass(self.T, self.Tprime)

        forcing_blocks = []
        for i, vhati in enumerate(vhatrs):
            f_row = zero()
            for j, f in enumerate(fs):
                f_row += Constant(self.dt) * Constant(P_M[j, i]) * replace(f, {self.test: vhati})
            forcing_blocks.append(f_row)
        
        return forcing_blocks

    # Helper function to expand all terms of u on the timestep boundaries
    def _expand_boundary_term(self, vhatrs, db, weak_bottom_u=None, weak_top_u=None):
        split_db = split_time_form_on(db, self.test)
        if split_db[0] != zero():
            raise AssertionError("Term not linear in test function found in time boundary term")
        split_db = split_db[1:]

        if weak_bottom_u:
            bottom_u = weak_bottom_u
        else:
            bottom_u = split(self.stage_mappings[self.trial])[0]
        
        if weak_top_u:
            top_u = weak_top_u
        else:
            top_u = split(self.stage_mappings[self.trial])[-1]

        # Compute maximum order of trial function in boundary form
        u_order = max_time_order(db, self.trial)

        # Tabulate (including derivatives) at the timestep endpoints for the trial function
        Tfs = self.T.finat_element.fiat_equivalent
        tab = Tfs.tabulate(u_order, [0., 1.])

        # Tabulate for the test function
        # bottom_v = split(self.stage_mappings[self.test])[1]
        # top_v = split(self.stage_mappings[self.test])[-1]
        bottom_v = vhatrs[0]
        top_v = vhatrs[-1]
        Tprimefs = self.T.finat_element.fiat_equivalent
        tabprime = Tprimefs.tabulate(len(split_db), [0., 1.])

        # Configure the correct replacements
        bottom_point_repl = {}
        top_point_repl = {}
        for n in reversed(range(u_order + 1)):
            dt_expr = dt_n(self.trial, n)
            bottom_point_repl[dt_expr] = Constant((1/self.dt)**n) * Constant(tab[(n,)][0,0]) * bottom_u
            top_point_repl[dt_expr] = Constant((1/self.dt)**n) * Constant(tab[(n,)][-1,-1]) * top_u
        bottom_point_repl[self.test] = bottom_v
        top_point_repl[self.test] = top_v
        
        bottom_db = zero()
        top_db = zero()
        for n_v, form in enumerate(split_db):
            bottom_db += Constant(-(1/self.dt)**n_v) * Constant(tabprime[(n_v,)][0,0]) * replace(form, bottom_point_repl)
            top_db += Constant((1/self.dt)**n_v) * Constant(tabprime[(n_v,)][-1,-1]) * replace(form, top_point_repl)

        return bottom_db, top_db


# Time form generator that generates forms for a problem according to the cPG (continuous Petrov-Galerkin)
# specification for variational timestepping.
class PetrovTimeFormGenerator(SplitTimeFormGenerator):
    def __init__(self, F, u, v, dt, time_quadrature, T, Tprime, stage_mappings):
        # Dimension of trial function space must be one more than test function space
        if T.dim() != Tprime.dim() + 1:
            raise AssertionError("Trial function space dimension must be one greater than "
                                 "test function space dimension for Petrov-Galerkin, but "
                                 "got dim(trial): " + str(T.dim() - 1) + " and "
                                 "dim(test): " + str(Tprime.dim() - 1))

        # Base initialisation
        super().__init__(F, u, v, dt, time_quadrature, T, Tprime, stage_mappings)

    def make_timestep_forms(self, u0, fs=[], db=zero()):
        # Number of time stages for trial/test functions and rhs values must be the same
        if fs and len(split(self.stage_mappings[self.trial])) != len(fs):
            raise AssertionError("Function time stages number does not correspond to RHS stages "
                                 "number, got #trial: " + str(len(split(self.stage_mappings[self.trial]))) + ", "
                                 "#rhs: " + str(len(fs)))
        total_form = zero()
        total_rhs_form = zero()

        # Make initial condition forms
        uhat0 = split(self.stage_mappings[self.trial])[0]
        vhat0 = split(self.stage_mappings[self.test])[0]
        ic_form = uhat0 * vhat0 * dx
        rho_form = u0 * vhat0 * dx

        total_form += ic_form
        total_rhs_form += rho_form

        # Filter the test functions that will solve the Petrov-Galerkin problem in the
        # rest of the timestep
        vhatrs = split(self.stage_mappings[self.test])[1:]

        # Add interior timestep forms
        interior_forms = self._compose_interior_form_block(vhatrs)
        for form in interior_forms:
            total_form += form 
        
        # Add the time boundary contributions if present
        if db:
            expanded_db_bottom, expanded_db_top = self._expand_boundary_term(vhatrs, db)
            total_form += expanded_db_bottom
            total_form += expanded_db_top
                
        # Add the contributions from the forcing function
        rhs_form = zero()
        forcing_forms = self._compose_forcing_form_block(vhatrs, fs)
        for form in forcing_forms:
            rhs_form += form
        total_rhs_form += rhs_form

        # Return total lhs and rhs forms
        return (total_form, total_rhs_form)


# Time form generator that generates forms for a problem according to the DG (discontinuous Galerkin)
# specification for variational timestepping. Uses simple time-directed upwinding.
# Splits form on the test function and substitutes the mixed space expansions for the trial function u
class DiscontinuousTimeFormGenerator(SplitTimeFormGenerator):
    def __init__(self, F, u, v, dt, time_quadrature, T, stage_mappings):
        # Base initialisation, note same trial and test function space
        super().__init__(F, u, v, dt, time_quadrature, T, T, stage_mappings)


    def make_timestep_forms(self, u0, fs=[], db=zero()):
        # Number of time stages for trial/test functions and rhs values must be the same
        if fs and len(split(self.stage_mappings[self.trial])) != len(fs):
            raise AssertionError("Function time stages number does not correspond to RHS stages "
                                 "number, got #trial: " + str(len(split(self.stage_mappings[self.trial]))) + ", "
                                 "#rhs: " + str(len(fs)))
        total_form = zero()
        total_rhs_form = zero()

        # For DG, no test function explicitly fixes the initial condition
        vhatrs = split(self.stage_mappings[self.test])

        # Add interior timestep forms
        interior_forms = self._compose_interior_form_block(vhatrs)
        for form in interior_forms:
            total_form += form 
        
        # Add the time boundary contributions if present
        if db:
            expanded_db_bottom, expanded_db_top = self._expand_boundary_term(vhatrs, db, weak_bottom_u=u0)
            total_form += expanded_db_bottom
            total_form += expanded_db_top
                
        # Add the contributions from the forcing function
        rhs_form = zero()
        forcing_forms = self._compose_forcing_form_block(vhatrs, fs)
        for form in forcing_forms:
            rhs_form += form
        total_rhs_form += rhs_form

        # Return total lhs and rhs forms
        return (total_form, total_rhs_form)



# Class that generates forms for a problem according to the cPG (continuous Petrov-Galerkin)
# specification for variational timestepping
class OldPetrovTimeFormGenerator(TimeFormGenerator):
    def __init__(self, F, u, v, dt, time_quadrature, T, Tprime, stage_mappings):
        # Dimension of trial function space must be one more than test function space
        if T.dim() != Tprime.dim() + 1:
            raise AssertionError("Trial function space dimension must be one greater than "
                                 "test function space dimension for Petrov-Galerkin, but "
                                 "got dim(trial): " + str(T.dim() - 1) + " and "
                                 "dim(test): " + str(Tprime.dim() - 1))
        
        # Base initialisation
        super().__init__(F, u, v, dt, time_quadrature, T, Tprime, stage_mappings)

        # Generator specific initialisation
        self.splitted = split_time_form(F)

        self.P_M = time_quadrature.time_mass(T, Tprime)
        self.P_L = time_quadrature.time_half_stiffness_on_trial(T, Tprime)
        self.P_K = time_quadrature.time_stiffness(T, Tprime)

    # Generate the interior Petrov-Galerkin forms
    def make_intermediate_forms(self, fs=[]):
        # Number of time stages for trial/test functions and rhs values must be the same
        if fs == []:
            fs = [Constant(0) * self.test * dx for _ in range(len(self.stage_mappings[self.trial]))]
        if len(self.stage_mappings[self.trial]) != len(fs):
            raise AssertionError("Function time stages number does not correspond to RHS stages "
                                 "number, got #trial: " + len(self.stage_mappings[self.trial]) + ", "
                                 "#rhs: " + len(fs))

        vhatis = self.stage_mappings[self.test][1:]

        clean_splitted = [strip_dt_form(order) for order in self.splitted]

        interior_form = zero()
        for i, vhati in enumerate(vhatis):
            for j, uhatj in enumerate(self.stage_mappings[self.trial]):
                repl_dict = {self.trial: uhatj, self.test:vhati}
                interior_form += self.P_L[j,i] * replace(clean_splitted[1], repl_dict)
                interior_form += self.dt * self.P_M[j,i] * replace(clean_splitted[0], repl_dict)

        rhs_form = zero()
        for i, vhati in enumerate(vhatis):
            f_row = zero()
            for j, f in enumerate(fs):
                f_row += self.dt * self.P_M[j, i] * replace(f, {self.test: vhati})
            rhs_form += f_row

        return (interior_form, rhs_form)