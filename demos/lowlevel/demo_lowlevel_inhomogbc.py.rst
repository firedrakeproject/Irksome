Time-dependent BCs and the low-level interface
==============================================

This demo shows how to update inhomogeneous and time-dependent
boundary conditions with the low-level interface.  We are using a
different manufactured solution than before, which is obvious from
reading through the code.

Imports::
  
  from firedrake import *  
  from ufl.algorithms.ad import expand_derivatives
 
  from irksome import GaussLegendre, getForm, Dt
  
  butcher_tableau = GaussLegendre(2)
  N = 64

  dt = Constant(10 / N)
  t = Constant(0.0)
  
  msh = UnitSquareMesh(N, N)
 
  V = FunctionSpace(msh, "CG", 1)
  x, y = SpatialCoordinate(msh)

  uexact = exp(-t) * cos(pi * x) * sin(pi * y)
  rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

  u = interpolate(uexact, V)

  v = TestFunction(V)
  F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx

  bc = DirichletBC(V, uexact, "on_boundary")

As with the homogeneous BC case, we use the `getForm` method to
process the semidiscrete problem::

  Fnew, k, bcnew, bcdata = getForm(F, butcher_tableau, t, dt, u, bcs=bc)

Recall that `getForm` produces:

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

We just use basic solver parameters and set up the variational problem
and solver::

  luparams = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

  prob = NonlinearVariationalProblem(Fnew, k, bcs=bcnew)
  solver = NonlinearVariationalSolver(prob, solver_parameters=luparams)

  ks = k.split()

Now, our time-stepping loop shows how to manually update the per-stage
boundary conditions at each time step::

  while (float(t) < 1.0):
      if float(t) + float(dt) > 1.0:
          dt.assign(1.0 - float(t))

      for (gdat, gcur) in bcdata:
          gdat.interpolate(gcur)

      solver.solve()

      for i in range(butcher_tableau.num_stages):
          u += float(dt) * butcher_tableau.b[i] * ks[i]

      t.assign(float(t) + float(dt))
      print(float(t))

  print()
  print(errornorm(uexact, u)/norm(uexact))
