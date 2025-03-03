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

    (u_t, v) + (\nabla u, \nabla v) = (f, v)\quad \forall v \in V \textrm{ and } t\in [0, T],

subject to the boundary condition :math:`u = g` on :math:`\Gamma`.  This demo uses particular choices of the 
functions :math:`f` and :math:`g` to be defined below.

The approach to bounds constraints below relies on the geometric properties of the Bernstein basis. 
In one dimension (on :math:`[0,1]`), the graph of the polynomial 
.. math::

   p(x) = \sum_{i = 0}^n p_i b_i^n(x),

where :math:`b_i^n(x)` are Bernstein basis polynomials, lies in the convex-hull of the points
.. math::

   \left\{\left(\frac{i}{n}, p_i\right)\right\}_{i = 0}^n.

In particular, if the coefficients :math:`p_i` lie in the interval :math:`[m,M]`, then the output of :math:`p(x)` will 
also fall within this range.  Similar results hold in higher dimensions.  This property provides a straightforward 
approach to uniformly enforced bounds constraints in both space and time.

First, we must import firedrake and certain items from Irksome: ::

    from firedrake import *
    from irksome import Dt, MeshConstant, RadauIIA, TimeStepper, BoundsConstrainedDirichletBC

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
        "mat_type": "aij",
        "pc_type": "lu"
    }

    vi_params = {
        "snes_type": "vinewtonrsls",
        "snes_max_it": 300,
        "snes_atol": 1.e-8,
        "ksp_type": "preonly",
        "mat_type": "aij",
        "pc_type": "lu",
    }


We now define the right-hand side using the method of manufactured solutions: ::

    x, y = SpatialCoordinate(msh)

    uexact = 0.5 * exp(-t) * (1 + (tanh((0.1 - sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)) / 0.015)))

    rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

Note that the exact solution is uniformly positive in space and time. Using a manufactured 
solution, one usually interpolates or projects the exact solution at time :math:`t = 0` onto the 
approximation space to obtain the initial condition. Interpolation does not work with the 
Bernstein basis, and there is no guarantee that an interpolant or projection would satisfy the bounds constraints. 
To guarantee that the initial condition satisfies the bounds constraints, we solve a variational 
inequality: ::

    v = TestFunction(V)
    u_init = Function(V)

    G = inner(u_init - uexact, v) * dx

    nlvp = NonlinearVariationalProblem(G, u_init)
    nlvs = NonlinearVariationalSolver(nlvp, solver_parameters=vi_params)

    lb = Function(V)
    ub = Function(V)

    ub.assign(np.inf)
    lb.assign(0.0)

    nlvs.solve(bounds=(lb, ub))

    u = Function(V)
    u.assign(u_init)

    u_c = Function(V)
    u_c.assign(u_init)

``u`` and ``u_c`` now hold a bounds-constrained approximation to the exact solution 
at :math:`t = 0`.  Note that `ub = None` is also supported and gets internally converted
to what we have here.

We now construct semidiscrete variational problems for both the constrained and unconstrained 
approximations using UFL notation and the ``Dt`` operator from Irksome: ::

    v = TestFunction(V)

    F = (inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx - inner(rhs, v) * dx)

    v_c = TestFunction(V)

    F_c = (inner(Dt(u_c), v_c) * dx + inner(grad(u_c), grad(v_c)) * dx - inner(rhs, v_c) * dx)

We use exact boundary conditions in both cases. When :math:`g` is the trace of a function 
defined over the whole domain, Firedrake creates its own version of the boundary condition by either interpolating 
or projecting that function onto the finite element space and computing the trace of the result. 
To ensure the internal boundary condition satisfies the bounds constraints, we will pass the bounds to 
the :class:`TimeStepper` below. ::

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
as ``solver_parameters`` to the :class:`.TimeStepper`.

We set the bounds as follows (reusing those defined in the initial condition): ::

    bounds = ('stage', lb, ub)

Internally, Firedrake will project the boundary condition expression into the entire space and match degrees of freedom
on the boundary.  This could introduce bounds violations.  To ensure this does not happen, we can use a special kind
of boundary condition that projects with bounds contraints. ::

    bc = BoundsConstrainedDirichletBC(V, uexact, "on_boundary", (lb, ub), solver_parameters=vi_params)

    kwargs_c = {"bounds": bounds,
                "stage_type": "value",
                "basis_type": 'Bernstein',
                "solver_parameters": vi_params
            }

    stepper_c = TimeStepper(F_c, butcher_tableau, t, dt, u_c, bcs=bc, **kwargs_c)

Note that if one does not set the ``basis_type`` to Bernstein, the standard basis will be used. Solving for the 
Bernstein coefficients of the collocation polynomial we obtain uniform-in-time bounds constraints. If the standard 
basis is used, the bounds constraints are guaranteed at the Runge-Kutta stages and the discrete times, but not necessarily 
between them.


When using a stage-value formulation, passing ``bounds`` to the :class:`TimeStepper` through the :meth:`~.TimeStepper.advance` method 
will enforce the bounds constraints at the discrete stages and time levels (this results in uniformly enforced constraints when using 
the Bernstein basis).

We now advance both semidiscrete systems in the usual way. We add the bounds as an argument 
to the :meth:`~.TimeStepper.advance` method for the constrained approximation.  

In order to monitor our approximate solutions, we check the minimum value of each after every step in time. 
If an approximate solution violates the lower bound, we append a tuple to indicate the time and minimum value. ::

    violations_for_unconstrained_method = []
    violations_for_constrained_method = []

    timestep = 0
    while (float(t) < float(Tf)):

        if (float(t) + float(dt) > float(Tf)):
            dt.assign(float(Tf) - float(t))

        stepper.advance()
        stepper_c.advance()

        t.assign(float(t) + float(dt))
        timestep = timestep + 1

        min_value = min(u.dat.data)
        if min_value < 0:
            violations_for_unconstrained_method.append((float(t), timestep, round(min_value, 3)))

        min_value_c = min(u_c.dat.data)
        if min_value_c < 0:
            violations_for_constrained_method.append((float(t), timestep, round(min_value_c, 3)))

        print(float(t))
  
Finally, we print the relative :math:`L^2` error and the time and severity (if any) of constraint violations: ::

    np.set_printoptions(legacy='1.25')

    print()
    print(f"Relative L^2 norm of the unconstrained solution: {norm(u - uexact) / norm(uexact)}")
    print(f"Relative L^2 norm of the constrained solution:   {norm(u_c - uexact) / norm(uexact)}")
    print()
    print("List of constraint violations in the form (time, time step, minimum value) for each approximation:")
    print()
    print(f"Unconstrained solution: {violations_for_unconstrained_method}")
    print()
    print(f"Constrained solution: {violations_for_constrained_method}")
