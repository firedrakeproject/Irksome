Block diagonal preconditioners for the heat equation
====================================================

This demo applies the method suggested in:

Mardal, Nilssen, Staff, "Order-optimal preconditioners for implicit
Runge-Kutta schemes applied to parabolic PDEs", SISC 29(1): 361--375 (2007),

to our ongoing heat equation demonstration problem on :math:`\Omega = [0,10]
\times [0,10]`, with boundary :math:`\Gamma`, giving rise to the weak form

.. math::

   (u_t, v) + (\nabla u, \nabla v) = (f, v)

A multi-stage RK method applied to the heat equation gives a
block-structured system.  The on-diagonal blocks are quite similar to
what one obtains from a backward Euler discretization of the equation.

With a 2-stage method, we have

.. math::
   
   \left[ \begin{array}{cc} A_{11} & A_{12} \\ A_{21} & A_{22} \end{array} \right]
   \left[ \begin{array}{c} k_1 \\ k_2 \end{array} \right]
   &= \left[ \begin{array}{c} f_1 \\ f_2 \end{array} \right]

And the suggestion (analyzed rigorously) of Mardal, Nilssen, and Staff
is to use a block diagonal preconditioner:

.. math::

  P = \left[ \begin{array}{cc} A_{11} & 0 \\ 0 & A_{22} \end{array} \right]


This allows one to leverage an existing methodology for a low order
method like backward Euler for the diagonal blocks.  In our case, we
will simply use an algebraic multigrid scheme, although one could
certainly use geometric multigrid or some other technique.

Common set-up for the problem::

  from firedrake import *  # noqa: F403
  from ufl.algorithms.ad import expand_derivatives
  from irksome import LobattoIIIC, TimeStepper, Dt

  butcher_tableau = LobattoIIIC(3)

  N = 64

  x0 = 0.0
  x1 = 10.0
  y0 = 0.0
  y1 = 10.0

  msh = RectangleMesh(N, N, x1, y1)

  dt = Constant(10. / N)
  t = Constant(0.0)

  V = FunctionSpace(msh, "CG", 1)
  x, y = SpatialCoordinate(msh)

  S = Constant(2.0)
  C = Constant(1000.0)
  B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
  R = (x * x + y * y) ** 0.5

  uexact = B * atan(t)*(pi / 2.0 - atan(S * (R - t)))
  rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))
  u = Function(V)
  u.interpolate(uexact)
  v = TestFunction(V)
  F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx

  bc = DirichletBC(V, 0, "on_boundary")

Now, we define the solver parameters.  PETSc-speak for taking the
block diagonal is an "additive fieldsplit"::

  params = {"mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "gmres",
            "ksp_monitor": None,
            "pc_type": "fieldsplit",   # block preconditioner
            "pc_fieldsplit_type": "additive"  # block diagaonal
            }

We also have to configure the (approximate) inverse of for each
diagonal block.  We'll just apply a sweek of gamg (PETSC's algebraic
multigrid)::

  per_field = {"ksp_type": "preonly",
               "pc_type": "gamg"}

  for s in range(butcher_tableau.num_stages):
      params["fieldsplit_%s" % (s,)] = per_field

Note that we have used the same technique for each RK stage, which is
probably typical.  However, it is not necessary at all.
      
To test this preconditioning strategy, we'll create a time stepping
object which will set up the variational problem for us::

  stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                        solver_parameters=params)

But, since we're just testing the efficacy of the preconditioner,
we'll solve the inside variational problem one time::

  stepper.solver.solve()
