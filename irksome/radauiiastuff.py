# Applies RadauIIA methods (very special case!) in terms of the polynomial
# values at each interpolation point instead of in terms
# of the RK stages.  This makes BCs and additively partitioned methods
# easier to implement

from functools import reduce
from operator import mul

import numpy
from firedrake import (Constant, DirichletBC, Function, TestFunction, dx, inner,
                       interpolate, project, split, MixedVectorSpaceBasis)
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake.dmhooks import push_parent

from irksome.getForm import replace, ConstantOrZero
from ufl.classes import Zero
from irksome.deriv import TimeDerivative

from irksome.manipulation import extract_terms


def getForm(F, butch, t, dt, u0, bcs=None, nullspace=None):
    """Given a time-dependent variational form and a
    :class:`RadauIIA`, produce UFL for the s-stage RK method.
         in terms of the point values rather than stage values.
    :arg F: UFL form for the PDE
    :arg butch: the :class:`RadauIIA` the collocation tableau being used
    :arg t: a :class:`Constant` referring to the current time level.
         Any explicit time-dependence in F is included through t.
    :arg dt: a :class:`Constant` referring to the size of the current
         time step.
    :arg u0: a :class:`Function` referring to the state of
         the PDE system at time `t`
    :arg bcs: optionally, a :class:`DirichletBC` object (or iterable thereof)
         containing (possible time-dependent) boundary conditions imposed
         on the system.
    :arg nullspace: A list of tuples of the form (index, VSB) where
         index is an index into the function space associated with `u`
         and VSB is a :class: `firedrake.VectorSpaceBasis` instance to
         be passed to a `firedrake.MixedVectorSpaceBasis` over the
         larger space associated with the Runge-Kutta method

    On output, we return a tuple consisting of four parts:

       - Fnew, the variational :class:`Form` for the problem we solve
         at each time step
       - UU, the :class:`firedrake.Function` for storing the solution at the
         collocation times.
         It lives in a :class:`firedrake.FunctionSpace` corresponding to the
         s-way tensor product of the space on which the semidiscrete
         form lives.
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the stages,
       - 'nspnew', the :class:`firedrake.MixedVectorSpaceBasis` object
         that represents the nullspace of the coupled system
       - `gblah`, a list of tuples of the form (f, expr, method),
         where f is a :class:`firedrake.Function` and expr is a
         :class:`ufl.Expr`.  At each time step, each expr needs to be
         re-interpolated/projected onto the corresponding f in order
         for Firedrake to pick up that time-dependent boundary
         conditions need to be re-applied.
         not.
    """
    v = F.arguments()[0]
    V = v.function_space()
    assert V == u0.function_space()

    if bcs is None:
        bcs = []

    c = numpy.array([Constant(ci) for ci in butch.c],
                    dtype=object)

    A1 = numpy.array([[ConstantOrZero(aa) for aa in arow]
                      for arow in butch.A], dtype=object)

    num_stages = butch.num_stages
    num_fields = len(V)

    Vbig = reduce(mul, (V for _ in range(num_stages)))
    # Silence a warning about transfer managers when we
    # coarsen coefficients in V
    push_parent(V.dm, Vbig.dm)

    vnew = TestFunction(Vbig)
    UU = Function(Vbig)

    if len(V) == 1:
        u0bits = [u0]
        vbits = [v]
        if num_stages == 1:
            vbigbits = [vnew]
            UUbits = [UU]
        else:
            vbigbits = split(vnew)
            UUbits = split(UU)
    else:
        u0bits = split(u0)
        vbits = split(v)
        vbigbits = split(vnew)
        UUbits = split(UU)

    UUbits = numpy.reshape(
        numpy.asarray(UUbits), (num_stages, num_fields))
    vbigbits = numpy.reshape(
        numpy.asarray(vbigbits), (num_stages, num_fields))

    splt = extract_terms(F)

    # check time terms have Dt outermost.
    for intgrl in splt.time.integrals():
        assert isinstance(intgrl.integrand().ufl_operands[1], TimeDerivative)

    # Replace Dt(F(u0)) with difference between F(stage[i]) and F(u0),
    # Will multiply through the other terms by dt.
    Fnew = Zero()
    for j in range(num_stages):
        repl = {t: t + c[j] * dt}
        for k, (ubit, vbit) in enumerate(zip(u0bits, vbits)):
            for intgrl in splt.time.integrals():

        
    # Now for bcs.
    bcnew = []
    gblah = []

    for bc in bcs:
        sub = 0 if len(V) == 1 else bc.function_space_index()
        Vsp = V if len(V) == 1 else V.sub(sub)
        offset = lambda i: sub + num_fields * i
        for i in range(num_stages):
            g = bc._original_arg
            gnew = replace(g, {t: t + c[i] * dt})
            try:
                gdat = interpolate(gnew, Vsp)
                gmeth = lambda g: gdat.interpolate(g)
            except:  # noqa: E722
                gdat = project(gnew, Vsp)
                gmeth = lambda g: gdat.project(g)
            gblah.append((gdat, gnew, gmeth))
            bcnew.append(DirichletBC(Vbig[offset(i)],
                                     gdat,
                                     bc.sub_domain))

    if nullspace is None:
        nspnew = None
    else:
        try:
            nullspace.sort()
        except TypeError:
            raise TypeError("Nullspace entries must be of form (idx, VSP), where idx is a non-negative integer")
        if (nullspace[-1][0] > num_fields) or (nullspace[0][0] < 0):
            raise ValueError("At least one index for nullspaces is out of range")
        nspnew = []
        for i in range(num_stages):
            count = 0
            for j in range(num_fields):
                if j == nullspace[count][0]:
                    nspnew.append(nullspace[count][1])
                    count += 1
                else:
                    nspnew.append(Vbig.sub(j + num_fields * i))
        nspnew = MixedVectorSpaceBasis(Vbig, nspnew)

    return Fnew, UU, bcnew, gblah, nspnew


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
            current time step.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing
            the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the RK method.
    :arg bc_type: How to manipulate the strongly-enforced boundary
            conditions to derive the stage boundary conditions.
            Should be a string, either "DAE", which implements BCs as
            constraints in the style of a differential-algebraic
            equation, or "ODE", which takes the time derivative of the
            boundary data and evaluates this for the stage values
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    :arg nullspace: A list of tuples of the form (index, VSB) where
            index is an index into the function space associated with
            `u` and VSB is a :class: `firedrake.VectorSpaceBasis`
            instance to be passed to a
            `firedrake.MixedVectorSpaceBasis` over the larger space
            associated with the Runge-Kutta method
    """
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None, nullspace=None):
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = len(butcher_tableau.b)
        self.butcher_tableau = butcher_tableau

        bigF, U, bigBCs, bigBCdata, bigNSP = \
            getForm(F, butcher_tableau, t, dt, u0, bcs, nullspace)

        self.U = U
        self.Us = U.split()
        self.bigBCs = bigBCs
        self.bigBCdata = bigBCdata
        problem = NLVP(bigF, self.U, bigBCs)

        appctx = {"F": F,
                  "butcher_tableau": butcher_tableau,
                  "t": t,
                  "dt": dt,
                  "u0": u0,
                  "bcs": bcs,
                  "nullspace": nullspace}
        self.solver = NLVS(problem,
                           appctx=appctx,
                           solver_parameters=solver_parameters,
                           nullspace=bigNSP)

    def advance(self):
        """Advances the system from time `t` to time `t + dt`.
        Note: overwrites the value `u0`."""

        for gdat, gcur, gmethod in self.bigBCdata:
            gmethod(gcur)

        self.solver.solve()

        # the last shall be first.
        ns = self.num_stages
        nf = self.num_fields
        for i, u0d in enumerate(self.u0.dat):
            u0d.data[:] = self.Us[nf*(ns-1)+i].dat.data_ro
