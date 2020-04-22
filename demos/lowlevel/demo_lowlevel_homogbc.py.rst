Not using the TimeStepper interface for the heat equation
=========================================================

Invariably, somebody will have a (possible) compelling reason to avoid
the top-level interface.  This demo shows how one can do just that.
It will be sparsely documented except for the critical bits and should
only be read after perusing the more basic demos.

We're solving the same problem that is done in the heat equation demos.

Imports::
  
  from firedrake import *  
  from ufl.algorithms.ad import expand_derivatives
 
  from irksome import GaussLegendre, getForm, Dt

Note that we imported `getForm` rather than :class:`TimeStepper`.  That's the
lower-level function inside Irksome that manipulates UFL and boundary conditions.

Continuing::
  
  butcher_tableau = GaussLegendre(1)
  N = 64

  dt = Constant(10 / N)
  t = Constant(0.0)
  
  x0 = 0.0
  x1 = 10.0
  y0 = 0.0
  y1 = 10.0

  msh = RectangleMesh(N, N, x1, y1)
  V = FunctionSpace(msh, "CG", 1)
  x, y = SpatialCoordinate(msh)

  S = Constant(2.0)
  C = Constant(1000.0)

  B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
  R = (x * x + y * y) ** 0.5
  uexact = B * atan(t)*(pi / 2.0 - atan(S * (R - t)))

  rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

  u = interpolate(uexact, V)

  v = TestFunction(V)
  F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx

  bc = DirichletBC(V, 0, "on_boundary")

Now, we use the `getForm` method, which processes the semidiscrete problem::

  Fnew, k, bcnew, bcdata = getForm(F, butcher_tableau, t, dt, u, bcs=bc)

This returns several things:

* `Fnew` is the UFL variational form for the fully discrete method.
* `k` is a new :class:`firedrake.Function` for  holding all the
  stages.  It lives on the s-way product of the space on which the
  problem was originally posed
* `bcnew` is a list of new :class:`firedrake.DirichletBC` that need to
  be enforced on the variational problem for the stages
* `bcdata` contains information needed to update the boundary
  conditions.  It is a list of pairs of the form (`f`, `expr`), where
  `f` is a :class:`firedrake.Function` and `expr` is a
  :class:`ufl.Expr` for each of the Dirichlet boundary conditions.
  Because Firedrake isn't smart enough to detect that `t` changes in
  the expression for the boundary condition, we need to manually
  interpolate or project each `expr` onto the corresponding `f` at the
  beginning of each time step.  Firedrake will notice this change and
  re-apply the boundary conditions.  This hassle is easy to overlook
  (not needed in this demo with homogeneous BC) and part of the reason
  we recommend using the :class:`TimeStepper` interface that does this
  for you.

Solver parameters are just blunt-force LU.  Other options are surely possible::

  luparams = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

We can set up a new nonlinear variational problem and create a solver
for it in standard Firedrake fashion::
	      
  prob = NonlinearVariationalProblem(Fnew, k, bcs=bcnew)
  solver = NonlinearVariationalSolver(prob, solver_parameters=luparams)

We'll need to split the stage variable so that we can update the
solution after solving for the stages at each time step::

  ks = k.split()

And here is our time-stepping loop.  Note that unlike in the higher-level
interface examples, we have to manually update the solution::
  
  while (float(t) < 1.0):
      if float(t) + float(dt) > 1.0:
          dt.assign(1.0 - float(t))
      solver.solve()

      for i in range(butcher_tableau.num_stages):
          u += float(dt) * butcher_tableau.b[i] * ks[i]

      t.assign(float(t) + float(dt))
      print(float(t))

  print()
  print(errornorm(uexact, u)/norm(uexact))
