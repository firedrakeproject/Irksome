Not using the TimeStepper interface for the heat equation
=========================================================

Invariably, somebody will have a (possibly) compelling reason or
at least urgent desire to avoid the top-level interface.  This demo
shows how one can do just that.
It will be sparsely documented except for the critical bits and should
only be read after perusing the more basic demos.

We're solving the same problem that is done in the heat equation demos.

Imports::

  from firedrake import *
  from ufl.algorithms.ad import expand_derivatives

  from irksome import GaussLegendre, getForm, Dt, MeshConstant
  from irksome.tools import get_stage_space

Note that we imported :func:`.getForm` rather than :class:`.TimeStepper`.  That's the
lower-level function inside Irksome that manipulates UFL and boundary conditions.

Continuing::

  butcher_tableau = GaussLegendre(1)
  N = 64

  x0 = 0.0
  x1 = 10.0
  y0 = 0.0
  y1 = 10.0

  msh = RectangleMesh(N, N, x1, y1)
  V = FunctionSpace(msh, "CG", 1)
  x, y = SpatialCoordinate(msh)

  MC = MeshConstant(msh)
  dt = MC.Constant(10 / N)
  t = MC.Constant(0.0)
  
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

Get the function space for the stage-coupled problem and a function to hold the stages we're computing::

  Vbig = get_stage_space(V, butcher_tableau.num_stages)
  k = Function(Vbig)

Get the variational form and bcs for the stage-coupled variational problem::

  Fnew, bcnew = getForm(F, butcher_tableau, t, dt, u, k, bcs=bc)

This returns several things:

* ``Fnew`` is the UFL variational form for the fully discrete method.
* ``bcnew`` is a list of new :class:`~firedrake.bcs.DirichletBC` that need to
  be enforced on the variational problem for the stages


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

  ks = k.subfunctions

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
