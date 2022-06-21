from firedrake import *
from fetsome.fet.fetutils import extract_time_points, generator_codes
from fetsome.fet.formgenerator import PetrovTimeFormGenerator, DiscontinuousTimeFormGenerator
from fetsome.fet.timequadrature import estimate_gauss_time_quadrature, make_gauss_time_quadrature
from fetsome.fet.timequadrature import estimate_radau_time_quadrature, make_radau_time_quadrature
from fetsome.fet.fetapi import TimeDirichletBC

import sys
from numpy import abs

class VariationalTimeStepper:
    def __init__(self, F, u0, u, v, T, t, dt, family="CPG", bcs=[], solver_parameters=None, time_quadrature=None):
        # Check that initial condition, trial and test functions belong to the same space
        if u.function_space() != u0.function_space() or u.function_space() != v.function_space():
            raise AssertionError("Initial condition, trial and test functions must be from "
                                 "the same function space.")
        
        # Check that the time element is a Lagrange element
        if T.ufl_element().family() != "Lagrange":
            raise AssertionError("Time finite element must be a Lagrange element.")
        
        # Check that the timestep is correctly defined
        if abs(dt) <= sys.float_info.epsilon or dt < 0.0:
            raise AssertionError("Timestep cannot be zero or negative.")
        
        # Check that the linear forcing term form is independent of the trial function
        if F[2] and F[2] != replace(F[2], {u: Constant(pi)}):
            raise AssertionError("Linear forcing term is not independent of trial function.")

        # Check the correctness of the passed family
        if not family in generator_codes:
            raise AssertionError("Unknown or unsupported family type " + family + ".")
        
        # Produce a warning if the forcing function is time independent
        if F[2] and F[2] == replace(F[2], {t: Constant(pi)}):
            print("Warning: the provided forcing function is time-independent, if unexpected "
                  "this might mean that the forcing function has already been interpolated onto the "
                  "spatial function space reducing time to a constant")
        
        for bc in bcs:
            if isinstance(bc, TimeDirichletBC):
                g = bc.get_time_expr()
                if g == replace(g, {t: Constant(pi)}):
                    print("Warning: at least one time dependent Dirichlet boundary condition is time independent, "
                        "this might mean that the condition has already been interpolated onto the "
                        "spatial function space reducing time to a constant")
                    break


        # Store all the useful initialisation arguments
        self.b = F[0]
        self.db = F[1]
        self.L = F[2]

        self.u0 = u0
        self.trial = u
        self.test = v
        self.T = T
        self.t = t
        self.dt = dt
        self.bcs = bcs
        self.solver_parameters = solver_parameters

        # Initialise informative fields
        self.curr_u = u0
        self.curr_time = 0.0
        self.next_time = self.dt
        
        # Build the time test function space
        self.kt = self.T.ufl_element().degree()
        if family == "CPG":
            # Does not use the TimeFunctionSpace utility as you need to pass in the mesh
            if self.kt == 1:
                self.Tprime = FunctionSpace(self.T.mesh(), "DG", 0)
            else:
                self.Tprime = FunctionSpace(self.T.mesh(), "CG", self.kt - 1)
        else:
            self.Tprime = T
        
        # Initialise the time point that corresponds to each evaluation point of the
        # time element's nodal basis
        self.time_points = extract_time_points(self.T)

        # Prepare the automatically generated quadrature rules if none is passed
        if not time_quadrature:
            if family == "CPG":
                # Make Gauss-Legendre time quadrature object for continuous Petrov-Galerkin
                num_quadrature_points = max([estimate_gauss_time_quadrature(form, self.t, self.kt) for form in (self.b, self.db, self.L)])
                self.time_quadrature = make_gauss_time_quadrature(num_quadrature_points)
            elif family == "DG":
                # Make Radau time quadrature object for discontinuous Galerkin
                num_quadrature_points = max([estimate_radau_time_quadrature(form, self.t, self.kt) for form in (self.b, self.db, self.L)])
                self.time_quadrature = make_radau_time_quadrature(num_quadrature_points)
        else:
            self.time_quadrature = time_quadrature

        # Make the effective function spaces and stage mappings
        self.V = u.function_space()
        self.Vt = MixedFunctionSpace((self.V,) * (self.kt + 1))

        self.uhat = Function(self.Vt)
        vhat = TestFunction(self.Vt)
        self.stage_mappings = {self.trial: self.uhat, self.test: vhat}

        # Make the generator for the specific FET family
        if family == "CPG":
            self.generator = PetrovTimeFormGenerator(self.b, self.trial, self.test, self.dt, self.time_quadrature,
                                                     self.T, self.Tprime, self.stage_mappings)

        elif family == "DG":
            self.generator = DiscontinuousTimeFormGenerator(self.b, self.trial, self.test, self.dt, self.time_quadrature,
                                                            self.T, self.stage_mappings)

        else:
            raise AssertionError("No registered generator for supported family " + family)
        
    # Advance the solution over a single timestep and return the result
    # (can also be result at all intermediate FET nodes).
    def advance(self, info=False, include_substages=False):
        uhat0 = Function(self.Vt)
        for sub in uhat0.split():
            # The initial guess for each intermediate step is the initial condition
            sub.interpolate(self.curr_u)
        
        if include_substages:
            us = []

        # Compose the forcing function linear forms for the intermediate steps
        if info:
            print("Composing RHS forcing term time forms.")
        fs = self._make_curr_forcing_forms()

        # Compose the spatial Dirichlet boundary conditions for intermediate steps
        bcs = self._make_curr_dirichlet_bcs()

        # Compose initial and intermediate forms
        if info:
            print("Composing block time forms.")
        total_lhs, total_rhs = self.generator.make_timestep_forms(split(uhat0)[0], fs=fs, db=self.db)
        F = total_lhs - total_rhs

        # Solve using the passed PETSc solver parameters (if any)
        if info:
            print("About to solve problem for timestep", self.curr_time, "->", self.next_time)

        if self.solver_parameters:
            solve(F == 0, self.uhat, bcs=bcs, solver_parameters=self.solver_parameters)
        else:
            solve(F == 0, self.uhat, bcs=bcs)
        
        if info:
            print("Solved.")

        # Add all nodes in the element to the solution if requested
        if include_substages:
            us.extend([self.uhat.sub(j).copy(deepcopy=True) for j in range(0, self.kt+1)])

        # Update the current value of the solution for the next iteration and timestep data
        self.curr_u = self.uhat.sub(self.kt)
        self.curr_time = self.next_time
        self.next_time += self.dt

        # Return the requested kind of solution (including or excluding substages)
        if include_substages:
            return us
        else:
            return self.curr_u

    # Returns the time that corresponds to the current held value of u
    def time_now(self):
        return self.curr_time

    # Returns the time that will correspond to the next advanced value of u
    def time_next(self):
        return self.next_time
    
    # Evaluates forcing function form at the timepoints for the element and
    # collects them in order
    def _make_curr_forcing_forms(self):
        if self.L:
            return [replace(self.L, {self.t: Constant(self.curr_time + self.dt * tau)}) for tau in self.time_points]
        return []
    
    # Evaluates Dirichlet boundary conditions at the timepoints and collects them
    # in the mixed function space
    def _make_curr_dirichlet_bcs(self):
        bcs = []
        for bc in self.bcs:
            sub_domain = bc.sub_domain
            if isinstance(bc, TimeDirichletBC):
                # Replace the time variable in the bc with the correct time point
                g = bc.get_time_expr()
                g_at_timepoints = [replace(g, {self.t: Constant(self.curr_time + self.dt * tau)}) for tau in self.time_points]
                bcs += [DirichletBC(self.Vt.sub(i), g_at_i, sub_domain) for i, g_at_i in enumerate(g_at_timepoints)]
            else:
                # Repeat the boundary condition for each sub-function
                g = bc.function_arg
                bcs += [DirichletBC(self.Vt.sub(i), g, sub_domain) for i in range(len(self.time_points))]
        return bcs
