Solving the Heat Equation with Bounds Constraints
=================================================

In this demo we solve the simple heat equation with bounds constraints applied uniformly in time and space. 
Consider the simple heat equation on :math:`\Omega = [0,1]\times [0,1]`, with boundary :math:`\Gamma`:

.. math::

    u_t - \Delta u &= f

    u & = g \quad \textrm{on}\ \Gamma

for some known functions :math:`f` and :math:`g`. The solution will be some function :math:`u\in V`, for 
a suitable function space :math:`V`.

The weak form is found by multiplying by an arbitrary test function :math:`v\in V` and integrating over :math:`\Omega`. 
We then have the variational problem of finding :math:`u:[0,T]\rightarrow V` such that 
.. math::

    (u_t, v) + (\nabla u, \nabla v) = (f, v)

This demo uses particular choices of the functions :math:`f` and :math:`g` to be defined below.

First, we must import firedrake and certain items from Irksome: ::

    from firedrake import *
    from irksome import (Dt, MeshConstant, RadauIIA, TimeStepper)

We also need some UFL tools in order to manufacture a solution: ::

    from ufl.algorithms import expand_derivatives

Finally, numpy provides us with the upper bound of infinity: ::

    import numpy as np

We first define the mesh and the necessary function space. We choose 
quadratic Bernstein polynomials to support bounds-constraints in space: ::

    N = 32

    msh = UnitSquareMesh(N, N)
    V = FunctionSpace(msh, "Bernstein", 2)

In order to enforce bounds constraints in time, we must utilize a collocation method. 
In this demo, we will time-step using the L-stable, fully implicit, 2-stage RadauIIA 
Runge-Kutta method. The bounds will be passed as an argument to the 
:meth:`~.TimeStepper.advance` method. We now define the Butcher Tableau and variables to store the 
time step, current time, and final time: ::

    butcher_tableau = RadauIIA(2)

    MC = MeshConstant(msh)
    dt = MC.Constant(2 / N)
    t = MC.Constant(0.0)
    Tf = MC.Constant(1.0)

We will find an approximate solution at time :math:`t=1.0` with and without 
enforcing a constraint on the lower bound. We will need the following pair of solver 
parameters: ::

    lu_params = {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "mat_typ": "aij",
        "pc_type": "lu"
    }

    vi_params = {
        "snes_type": "vinewtonrsls",
        "snes_max_it": 300,
        "snes_atol": 1.e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }


We now define the right-hand side using the method of manufactured solutions: ::

    x, y = SpatialCoordinate(msh)

    uexact = 0.5 * exp(-t) * (1 + (tanh((0.1 - sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)) / 0.015)))

    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

Note that the exact solution is uniformly positive in space and time. Using a manufactured 
solution, one would typically project the exact solution at time :math:`t = 0` onto the 
approximation space and use the projection as the initial condition. In this case, the 
projection does not satisfy the lower bound of :math:`0`. We instead solve a variational inequality 
to find a bounds-preserving initial condition: ::

    v = TestFunction(V)
    u = Function(V)

    G = inner(u - uexact, v) * dx

    nlvp = NonlinearVariationalProblem(G, u)
    nlvs = NonlinearVariationalSolver(nlvp, solver_parameters=vi_params)

    upper = Function(V)
    lower = Function(V)

    with upper.dat.vec as upper_vec:
        upper_vec.set(np.inf)

    with lower.dat.vec as lower_vec:
        lower_vec.set(0.0)

    nlvs.solve(bounds=(lower, upper))

    u_c = u.copy(deepcopy=True)

``u`` and ``u_c`` now hold a bounds-constrained approximation to the exact solution 
at :math:`t = 0`.

We now construct semidiscrete variational problems for both the constrained and unconstrained 
approximations using UFL notation and the ``Dt`` operator from Irksome: ::

    v = TestFunction(V)

    F = (inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - inner(rhs, v) * dx)

    v_c = TestFunction(V)

    F_c = (inner(Dt(u_c), v_c) * dx + inner(grad(u_c), grad(v_c)) * dx - inner(rhs, v_c) * dx)

We use exact boundary conditions in both cases: ::

    bc = DirichletBC(V, uexact, "on_boundary")

For the unconstrained approximation, we configure the :class:`.TimeStepper` in a 
familiar way: ::

    stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc, solver_parameters=lu_params)

We will enforce nonnegativity when finding the constrained approximation. We now set up the keyword database to 
configure an instance of :class:`.TimeStepper` for this task. We first specify, using the 
keyword ``stage_type``, that we wish to use a stage-value formulation of the underlying collocation 
method. The keyword ``basis_type`` then allows us to change the basis of the collocation 
polynomial to the Bernstein basis. Having done this, we must specify a solver which is able to handle bounds 
constraints. In this example we solve a variational inequality using ``vinewtonrsls`` by passing ``vi_params`` 
as ``solver_parameters`` to the :class:`.TimeStepper`: ::

    kwargs_c = {"stage_type": "value",
                "basis_type": 'Bernstein',
                "solver_parameters": vi_params
            }

    stepper_c = TimeStepper(F_c, butcher_tableau, t, dt, u_c, bcs=bc, **kwargs_c)

We set the bounds as follows: ::

    lb = Function(V)
    lb.assign(0)

    ub = None

    bounds = ('stage', lb, ub)

Passing the :meth:`~.TimeStepper.advance` method ``bounds`` will enforce the 
given bounds on the coefficients of the collocation polynomial (now represented in the Bernstein basis). This 
enforces the lower bound of :math:`0` uniformly in time.

We now advance both semidiscrete systems in the usual way. We add the bounds as an argument 
to the :meth:`~.TimeStepper.advance` method for the constrained approximation.  

In order to monitor our approximate solutions, we check the minimum value of each after every step in time. 
If an approximate solution violates the lower bound, we append a tuple to indicate the time and minimum value. ::

    mins = []
    mins_c = []

    while (float(t) < float(Tf)):

        if (float(t) + float(dt) > float(Tf)):
            dt.assign(float(Tf) - float(t))

        stepper.advance()
        stepper_c.advance(bounds=bounds)

        t.assign(float(t) + float(dt))

        minv = min(u.dat.data)
        if minv < 0:
            mins.append((float(t), minv))

        minv_c = min(u_c.dat.data)
        if minv_c < 0:
            mins_c.append((float(t), minv_c))

        print(float(t))
  
Finally, we print the relative :math:`L^2` error and the time and severity (if any) of constraint violations: ::

    np.set_printoptions(legacy='1.25')

    print()
    print(norm(u - uexact) / norm(uexact))
    print(norm(u_c - uexact) / norm(uexact))
    print()
    print(mins)
    print(mins_c)