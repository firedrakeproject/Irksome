from .getForm import getForm
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake import Function, norm, VectorFunctionSpace
from ufl import VectorElement
import numpy


class TimeStepper:
    """Front-end class for advancing a time-dependent PDE via a Runge-Kutta
    method.

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
            the Runge-Kutta method to be used for time marching.
    :arg t: A :class:`firedrake.Constant` instance that always
            contains the time value at the beginning of a time step
    :arg dt: A :class:`firedrake.Constant` containing the size of the
            current time step.  The user may adjust this value between
            time steps, but see :class:`AdaptiveTimeStepper` for a
            method that attempts to do this automatically.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing
            the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the RK method.
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    """
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None):
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = len(butcher_tableau.b)
        self.butcher_tableau = butcher_tableau

        bigF, stages, bigBCs, bigBCdata = \
            getForm(F, butcher_tableau, t, dt, u0, bcs)

        self.stages = stages
        self.bigBCs = bigBCs
        self.bigBCdata = bigBCdata
        problem = NLVP(bigF, stages, bigBCs)
        self.solver = NLVS(problem, solver_parameters=solver_parameters)

        self.ks = stages.split()

    def _update(self):
        """Assuming the algebraic problem for the RK stages has been
        solved, updates the solution.  This will not typically be
        called by an end user."""
        b = self.butcher_tableau.b
        dtc = float(self.dt)
        u0 = self.u0
        ns = self.num_stages
        nf = self.num_fields

        if isinstance(self.stages.function_space().ufl_element(),
                      VectorElement):
            u0.dat.data[:] += dtc * numpy.dot(self.stages.dat.data, b)
        elif nf == 1:
            ks = self.ks
            for i in range(ns):
                u0 += dtc * b[i] * ks[i]
        else:
            k = self.stages

            for s in range(ns):
                for i in range(nf):
                    u0.dat.data[i][:] += dtc * b[s] * k.dat.data[nf * s + i][:]

    def advance(self):
        """Advances the system from time `t` to time `t + dt`.
        Note: overwrites the value `u0`."""
        for gdat, gcur in self.bigBCdata:
            gdat.interpolate(gcur)

        self.solver.solve()

        self._update()


class AdaptiveTimeStepper(TimeStepper):
    """Front-end class for advancing a time-dependent PDE via a Runge-Kutta
    method.

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
            the Runge-Kutta method to be used for time marching.
    :arg t: A :class:`firedrake.Constant` instance that always
            contains the time value at the beginning of a time step
    :arg dt: A :class:`firedrake.Constant` containing the size of the
            current time step.  The user may adjust this value between
            time steps, but see :class:`AdaptiveTimeStepper` for a
            method that attempts to do this automatically.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg tol: The temporal ttruncation error tolerance
    :arg dtmin: Minimal acceptable time step.  An exception is raised
            if the step size drops below this threshhold.
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing
            the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the RK method.
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    """
    def __init__(self, F, butcher_tableau, t, dt, u0,
                 tol=1.e-6, dtmin=1.e-5, bcs=None, solver_parameters=None):
        assert butcher_tableau.btilde is not None
        super(AdaptiveTimeStepper, self).__init__(F, butcher_tableau,
                                                  t, dt, u0, bcs,
                                                  solver_parameters)
        self.tol = tol
        self.dt_min = dtmin
        self.delb = butcher_tableau.b - butcher_tableau.btilde
        self.error_func = Function(u0.function_space())

    def _estimate_error(self):
        """Assuming that the RK stages have been evaluated, estimates
        the temporal truncation error by taking the norm of the
        difference between the new solutions computed by the two
        methods.  Typically will not be called by the end user."""
        dtc = float(self.dt)
        delb = self.delb

        if self.num_fields == 1:
            ks = self.ks
            self.error_func.dat.data[:] = 0.0
            for i in range(self.num_stages):
                self.error_func += dtc * delb[i] * ks[i]
        else:
            k = self.stages
            for i in range(self.num_fields):
                self.error_func.dat.data[i][:] = 0.0
            for s in range(self.num_stages):
                for i in range(self.num_fields):
                    self.error_func.dat.data[i][:] += \
                        dtc * delb[s] * k.dat.data[self.num_fields*s+i][:]

        return norm(self.error_func)

    def advance(self):
        """Attempts to advances the system from time `t` to time `t +
        dt`.  If the error threshhold is exceeded, will adaptively
        decrease the time step until the step is accepted.  Also
        predicts new time step once the step is accepted.

        Note: overwrites the value `u0`."""
        print("\tTrying dt=", float(self.dt))
        while 1:
            for gdat, gcur in self.bigBCdata:
                gdat.interpolate(gcur)

            self.solver.solve()
            err = self._estimate_error()

            print("\tTruncation error", err)
            q = 0.84 * (self.tol / err)**(1./(self.butcher_tableau.order-1))
            print("\tq factor:", q)
            if q <= 0.1:
                q = 0.1
            elif q >= 4.0:
                q = 4.0

            dtnew = q * float(self.dt)

            if err >= self.tol:
                print("\tShrinking time step to ", dtnew)
                self.dt.assign(dtnew)
            elif dtnew <= self.dt_min:
                raise RuntimeError("Minimum time step threshold violated")
            else:
                print("\tStep accepted, new time step is ", dtnew)
                self._update()
                self.dt.assign(dtnew)
                return (err, dtnew)
