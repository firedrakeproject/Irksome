Solving the heat equation with monolithic multigrid and Discontinuous Galerkin-in-Time
========================================================================

This reprise of the heat equation demo uses a monolithic multigrid
algorithm suggested by Patrick Farrell to perform time advancement,
but now applies it to the Discontinuous Galerkin-in-Time discretization.

We consider the heat equation on :math:`\Omega = [0,10]
\times [0,10]`, with boundary :math:`\Gamma`: giving rise to the weak form

.. math::

   (u_t, v) + (\nabla u, \nabla v) = (f, v)

This demo implements an example used by Solin with a particular choice
of :math:`f` given below

We perform similar imports and setup as before::

  from firedrake import *
  from irksome import DiscGalerkinTimeStepper, Dt, MeshConstant
  from ufl.algorithms.ad import expand_derivatives


However, we need to set up a mesh hierarchy to enable geometric multigrid
within Firedrake::

  N = 128
  x0 = 0.0
  x1 = 10.0
  y0 = 0.0
  y1 = 10.0

  from math import log
  coarseN = 8  # size of coarse grid
  nrefs = log(N/coarseN, 2)
  assert nrefs == int(nrefs)
  nrefs = int(nrefs)
  base = RectangleMesh(coarseN, coarseN, x1, y1)
  mh = MeshHierarchy(base, nrefs)
  msh = mh[-1]

From here, setting up the function space, manufactured solution, etc,
are just as for the regular heat equation demo::

  V = FunctionSpace(msh, "CG", 1)

  MC = MeshConstant(msh)
  dt = MC.Constant(10.0 / N)
  t = MC.Constant(0.0)

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

And now for the solver parameters.  Note that we are solving a
block-wise system with all stages coupled together.  This performs a
monolithic multigrid with pointwise block Jacobi preconditioning::

  mgparams = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "gmres",
              "ksp_monitor_true_residual": None,
              "pc_type": "mg",
              "mg_levels": {
                  "ksp_type": "chebyshev",
                  "ksp_max_it": 1,
                  "ksp_convergence_test": "skip",
                  "pc_type": "python",
                  "pc_python_type": "firedrake.PatchPC",
                  "patch": {
                      "pc_patch": {
                          "save_operators": True,
                          "partition_of_unity": False,
                          "construct_type": "star",
                          "construct_dim": 0,
                          "sub_mat_type": "seqdense",
                          "dense_inverse": True,
                          "precompute_element_tensors": None},
                      "sub": {
                          "ksp_type": "preonly",
                          "pc_type": "lu"}}},
              "mg_coarse": {
                  "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps"}
              }
  
These solver parameters work just fine in the :class:`.DiscGalerkinTimeStepper`::

  stepper = DiscGalerkinTimeStepper(F, 2, t, dt, u, bcs=bc,
                                    solver_parameters=mgparams)

And we can advance the solution in time in typical fashion::

  while (float(t) < 1.0):
      if (float(t) + float(dt) > 1.0):
          dt.assign(1.0 - float(t))
      stepper.advance()
      print(float(t), flush=True)
      t.assign(float(t) + float(dt))

After the solve, we can retrieve some statistics about the solver::

  steps, nonlinear_its, linear_its = stepper.solver_stats()

  print("Total number of timesteps was %d" % (steps))
  print("Average number of nonlinear iterations per timestep was %.2f" % (nonlinear_its/steps))
  print("Average number of linear iterations per timestep was %.2f" % (linear_its/steps))

Finally, we print out the relative :math:`L^2` error::

  print()
  print(norm(u-uexact)/norm(uexact))
