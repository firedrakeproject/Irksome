"""Classes to implement non-linearly partitioned Runge-Kutta (NPRK) schemes based on
Buvoli and Southworth."""
import numpy as np

from firedrake import Function, split, \
    NonlinearVariationalProblem, NonlinearVariationalSolver

from ufl.form import Form
from ufl.algorithms.analysis import extract_type

from .deriv import TimeDerivative
from .tools import MeshConstant, replace
from .manipulation import extract_terms
from .nprk_tableaux import NPRKTableau


def _has_len(x) -> bool:
    """Checks if an object has a length"""

    try:
        _ = len(x)
    except Exception:
        return False

    return True


def _check_aux_t_dep(F: Form, u0: Function, u1: Function, u2: Function) -> tuple[np.ndarray]:
    """Identifies which subfunctions are auxiliary (no time derivative)
    and which are time-dependent. Also checks that variables are exactly
    one of the two and that auxiliary variable do not appear in a partitioned
    argument."""

    num_fields = len(u0.function_space())

    if num_fields == 1:
        u0_bits = [u0]
        u1_bits = [u1]
        u2_bits = [u2]
    else:
        u0_bits = split(u0)
        u1_bits = split(u1)
        u2_bits = split(u2)

    # Gather all time derivatives
    derivs = extract_type(F, TimeDerivative)

    # Gather all variables outside of a time derivative
    types = {type(var) for var in u0_bits}
    F_no_dt = extract_terms(F).remainder
    vars_ = set().union(*(extract_type(F_no_dt, type_) for type_ in types))

    # Indices of variables which are auxiliary and time-dependent
    aux_list = np.array([], dtype=int)
    t_dep_list = np.array([], dtype=int)

    for i, (u0b, u1b, u2b) in enumerate(zip(u0_bits, u1_bits, u2_bits)):
        # Vector spaces need each component checked
        if _has_len(u0b):
            is_aux = (all(ub_ in vars_ for ub_ in u0b)
                      and all(TimeDerivative(ub_) not in derivs for ub_ in u0b))
            is_t_dep = (all(TimeDerivative(ub_) in derivs for ub_ in u0b)
                        and all(ub_ not in vars_ for ub_ in u0b))
            in_args = (any(ub_ in vars_ for ub_ in u1b)
                       and any(ub_ in vars_ for ub_ in u2b))
        else:
            is_aux = u0b in vars_ and TimeDerivative(u0b) not in derivs
            is_t_dep = TimeDerivative(u0b) in derivs and u0b not in vars_
            in_args = u1b in vars_ or u2b in vars_

        if is_aux and is_t_dep:
            raise RuntimeError(f"Variable {i} is both time-dependent and auxiliary.")

        if not (is_aux or is_t_dep):
            raise RuntimeError(f"Variable {i} is neither time-dependent nor auxiliary")

        if is_aux and in_args:
            raise RuntimeError(f"Variable {i} is auxiliary but used in an argument")

        if is_aux:
            aux_list = np.append(aux_list, i)

        if is_t_dep:
            t_dep_list = np.append(t_dep_list, i)

    return aux_list, t_dep_list


def getFormsNPRK(F, u0, u1, u2, t, dt):
    """Generates three forms: one for implicit solves in each of the arguments
    and one for mass solves"""

    V = F.arguments()[0].function_space()
    msh = V.mesh()
    MC = MeshConstant(msh)

    space_check = (V == u0.function_space()
                   == u1.function_space()
                   == u2.function_space())

    if not space_check:
        raise ValueError("All three functions and "
                         "Form argument must be on the same function space")

    # Stage value, implicitly solved for
    u_stage = Function(V)
    # Term formed from summing previous stages
    u_tilde = Function(V)
    # Argument kept explicit in implicit solves
    u_exp = Function(V)
    # Implicit coefficient
    a = MC.Constant(0.0)
    # Abscissae
    c = MC.Constant(0.0)

    # (NOTE): Replacing both u0 and Dt(u0) places stage values where auxiliary variables are
    #         and stage value updates where time derivatives are

    repl_1 = {u0: u_stage, TimeDerivative(u0): (u_stage - u_tilde)/(dt * a),
              u1: u_stage, u2: u_exp, t: t + c * dt}
    repl_2 = {u0: u_stage, TimeDerivative(u0): (u_stage - u_tilde)/(dt * a),
              u1: u_exp, u2: u_stage, t: t + c * dt}

    F_arg1 = replace(F, repl_1)
    F_arg2 = replace(F, repl_2)

    # Solution of explicit mass solve
    k = Function(V)
    # Functions for first and second argument of explicit mass solve
    u_arg1, u_arg2 = Function(V), Function(V)

    # (NOTE): Replacing both u0 and Dt(u0) allows the value of the auxiliary variables to be
    #         determined while solving for the stage derivative of time-dependent variables

    repl_mass = {u0: k, TimeDerivative(u0): k, u1: u_arg1, u2: u_arg2, t: t + c * dt}
    F_mass = replace(F, repl_mass)

    return u_stage, u_tilde, u_exp, (a, c), F_arg1, F_arg2, k, u_arg1, u_arg2, F_mass


class NPRKMethod:
    """Class for advancing a time-dependent PDE through a non-linearly partitioned
    Runge-Kutta (NPRK) method. This requires one form with three functions corresponding to
    the time derivatives/auxiliary terms, the first argument terms and the second argument terms.

    :arg F: A :class:`ufl.Form` instance describing partitioned operator F(t, u1, u2) == 0,
             where `u1` and `u2` denote the first and second arguments
    :arg nprk_tableau: A :class:`nprk_tableaux.NPRKTableau` instance giving the NPRK scheme
    :arg t: A :class:`MeshConstant` attribute referring to the current time
    :arg dt: A :class:`MeshConstant` attribute referring to the current time step
    :arg u0: A :class:`firedrake.Function` instance containing the current solution value.
             All :class:`irksome.deriv.TimeDerivative`s and auxiliary variables in `F` should
             be described in terms of `u0`
    :arg u1: A :class:`firedrake.Function` instance indicating the first argument in `F`
    :arg u2: A :class:`firedrake.Function` instance indicating the second argument in `F`
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing the strongly-enforced
              boundary conditions. These are applied at each stage of the RK method.
    :arg arg1_solver_parameters: A :class:`dict` of solver parameters that will be used in the
                                 implicit solves in the first argument.
    :arg arg2_solver_parameters: A :class:`dict` of solver parameters that will be used in the
                                 implicit solves in the second argument.
    :arg mass_parameters: A :class:`dict` of solver parameters that will be used in the mass
                          solves when the operator must be evaluated on previous stages.
    :arg nullspace: An optional null space object.
    """

    def __init__(self, F, nprk_tableau: NPRKTableau, t, dt, u0, u1, u2, bcs=None,
                 arg1_solver_parameters=None, arg2_solver_parameters=None,
                 mass_parameters=None, appctx=None, nullspace=None):

        self.num_steps = 0

        # Number of non-linear and linear iterations in solves in each argument
        self.num_nonlinear_iterations = np.array([0, 0], dtype=int)
        self.num_linear_iterations = np.array([0, 0], dtype=int)

        self.num_mass_nonlinear_iterations = 0
        self.num_mass_linear_iterations = 0

        self.tableau = nprk_tableau

        self.dt = dt
        self.u = u0

        # (NOTE): Can use list of time-dependent variables to reduce the length of the loops
        #         when summing over previous stages for a potential performance improvement
        _ = _check_aux_t_dep(F, u0, u1, u2)

        self.u_stage, self.u_tilde, self.u_exp, self.ac, F_arg1, F_arg2, \
            self.k, self.u_arg1, self.u_arg2, F_mass = getFormsNPRK(F, u0, u1, u2, t, dt)

        V = u0.function_space()
        # Operator evaluations for the pairs of stage arguments
        self.fs = {tuple(jk): Function(V) for jk in nprk_tableau.all_indices}
        # Stage values, needed for most general tableaux
        self.us = [Function(V) for _ in range(nprk_tableau.num_stages)]

        # BCs can be passed straight through to stage value solvers
        arg1_prob = NonlinearVariationalProblem(F_arg1, self.u_stage, bcs)
        arg2_prob = NonlinearVariationalProblem(F_arg2, self.u_stage, bcs)

        # Additional BCs for mass solvers
        bcs_mass = []
        for bc in bcs:
            bcs_mass.append(bc.reconstruct(g=0.0))

        mass_prob = NonlinearVariationalProblem(F_mass, self.k, bcs_mass)

        appctx_irksome = {"F": F,
                          "nprk_tableau": nprk_tableau,
                          "t": t,
                          "dt": dt,
                          "u0": u0,
                          "bcs": bcs,
                          "bc_type": "DAE",
                          "stage_type": "nprk",
                          "nullspace": nullspace}
        if appctx is None:
            appctx = appctx_irksome
        else:
            appctx = {**appctx, **appctx_irksome}

        # One solver corresponding to each implicit argument
        self.solvers = {}
        self.solvers[0] = NonlinearVariationalSolver(arg1_prob, appctx=appctx,
                                                     solver_parameters=arg1_solver_parameters,
                                                     nullspace=nullspace)
        self.solvers[1] = NonlinearVariationalSolver(arg2_prob, appctx=appctx,
                                                     solver_parameters=arg2_solver_parameters,
                                                     nullspace=nullspace)

        self.mass_solver = NonlinearVariationalSolver(mass_prob,
                                                      solver_parameters=mass_parameters,
                                                      nullspace=nullspace)

        if self.tableau.is_stiffly_accurate:
            self._finalize = self._finalize_stiffly_accurate
        else:
            self._finalize = self._finalize_general

    def advance(self):
        """Takes a single time step"""

        ns = self.tableau.num_stages
        args = self.tableau.impl_args
        coeffs = self.tableau.impl_coeffs
        impl_indices = self.tableau.impl_indices
        a_list = self.tableau.a_list
        exp_dict = self.tableau.expl_solve_dict
        abscissae = self.tableau.c_values

        a, c = self.ac

        u_stage = self.u_stage
        u_tilde = self.u_tilde
        fs = self.fs

        dtc = float(self.dt)

        self.u_stage.assign(self.u)
        self.us[0].assign(self.u_stage)

        # Loop over remaining stages
        for i in range(1, ns):

            # Calculate explicit operator evaluations from previous stage
            for jk in exp_dict.get(i-1, []):
                j_, k_ = jk[0], jk[1]
                self.u_arg1.assign(self.us[j_])
                self.u_arg2.assign(self.us[k_])
                self.mass_solver.solve()
                self.num_mass_nonlinear_iterations += self.mass_solver.snes.getIterationNumber()
                self.num_mass_linear_iterations += self.mass_solver.snes.getLinearSolveIterations()
                fs[tuple(jk)].assign(self.k)

            # Assign implicit coefficient and abscissae
            a.assign(coeffs[i-1])
            c.assign(abscissae[i-1])
            # Assign previous stage
            self.u_exp.assign(u_stage)

            # Sum over previous stages
            u_tilde.assign(self.u)
            a_indices, a_values = a_list[i-1]
            for jk, a_ijk in zip(a_indices, a_values):
                f_ = fs[tuple(jk)]
                for u_t_bit, fbit in zip(u_tilde.subfunctions, f_.subfunctions):
                    u_t_bit += dtc * float(a_ijk) * fbit

            # Implicit solve
            self.solvers[args[i-1]].solve()
            self.num_nonlinear_iterations[args[i-1]] += self.solvers[args[i-1]].snes.getIterationNumber()
            self.num_linear_iterations[args[i-1]] += self.solvers[args[i-1]].snes.getLinearSolveIterations()

            # Store stage value
            self.us[i].assign(u_stage)

            # Store operator evaluation in dictionary
            f_ = fs[tuple(impl_indices[i-1])]
            f_.assign((u_stage - u_tilde)/(dtc * a))

        self._finalize()
        self.num_steps += 1

    def _finalize_general(self):
        """Finalize by summing over stage derivatives"""
        b_indices = self.tableau.b_indices
        b_values = self.tableau.b_values
        u = self.u
        fs = self.fs
        dtc = float(self.dt)

        for jk, b_jk in zip(b_indices, b_values):
            f_ = fs[tuple(jk)]
            for ubit, fbit in zip(u.subfunctions, f_.subfunctions):
                ubit += dtc * float(b_jk) * fbit

    def _finalize_stiffly_accurate(self):
        """Finalize stiffly accurate case by assigning last stage"""
        self.u.assign(self.u_stage)

    def solver_stats(self):
        """Returns solver data"""
        return (self.num_steps,
                self.num_nonlinear_iterations,
                self.num_linear_iterations,
                self.num_mass_nonlinear_iterations,
                self.num_mass_linear_iterations)
