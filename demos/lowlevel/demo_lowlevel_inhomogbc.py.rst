Time-dependent BCs and the low-level interface
==============================================

This demo shows how to update inhomogeneous and time-dependent
boundary conditions with the low-level interface.  We are using a
different manufactured solution than before, which is obvious from
reading through the code.

Imports::

  from firedrake import *
  from ufl.algorithms.ad import expand_derivatives

  from irksome import GaussLegendre, getForm, Dt, MeshConstant
  
  butcher_tableau = GaussLegendre(2)
  N = 64

  msh = UnitSquareMesh(N, N)

  MC = MeshConstant(msh)
  dt = MC.Constant(10 / N)
  t = MC.Constant(0.0)
  
  V = FunctionSpace(msh, "CG", 1)
  x, y = SpatialCoordinate(msh)

  uexact = exp(-t) * cos(pi * x) * sin(pi * y)
  rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

  u = Function(V)
  u.interpolate(uexact)

  v = TestFunction(V)
  F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx

  bc = DirichletBC(V, uexact, "on_boundary")

As with the homogeneous BC case, we use the `getForm` method to
process the semidiscrete problem::

  Fnew, k, bcnew, nspnew, bcdata = getForm(F, butcher_tableau, t, dt, u, bcs=bc)

Recall that `getForm` produces:

* ``Fnew`` is the UFL variational form for the fully discrete method.
* ``k`` is a new :class:`~firedrake.function.Function` of stages on the s-way product of the space on which the problem was originally posed
* ``bcnew`` is a list of new :class:`~firedrake.bcs.DirichletBC` that need to
  be enforced on the variational problem for the stages
* ``bcdata`` contains information needed to update the boundary
  conditions.  It is a list of triples of the form
  (``f``,``expr``,``method``), where ``f`` is a
  :class:`~firedrake.function.Function`, ``expr`` is an
  :class:`~ufl.core.expr.Expr`, and ``method`` is either a project or
  interpolate operation for each of the Dirichlet boundary conditions.
  You're using the low-level interface and have to force Firedrake to
  reapply the boundary conditions.


We just use basic solver parameters and set up the variational problem
and solver::

  luparams = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

  prob = NonlinearVariationalProblem(Fnew, k, bcs=bcnew)
  solver = NonlinearVariationalSolver(prob, solver_parameters=luparams, nullspace=nspnew)

  ks = k.split()

Now, our time-stepping loop shows how to manually update the per-stage
boundary conditions at each time step::

  while (float(t) < 1.0):
      if float(t) + float(dt) > 1.0:
          dt.assign(1.0 - float(t))

      for (gdat, gcur, gmethod) in bcdata:
          gmethod(gcur, u)

      solver.solve()

      for i in range(butcher_tableau.num_stages):
          u += float(dt) * butcher_tableau.b[i] * ks[i]

      t.assign(float(t) + float(dt))
      print(float(t))

  print()
  print(errornorm(uexact, u)/norm(uexact))
