Solving the wave equation with monolithic multigrid
===================================================

This reprise of the heat equation demo uses a monolithic multigrid
algorithm to perform time advancement.

We consider the heat equation on :math:`\Omega = [0,1]
\times [0,1]`, with boundary :math:`\Gamma`: giving rise to the weak form

.. math::

   (u_{tt}, v) + (\nabla u, \nabla v) = 0

We perform similar imports and setup as before::

  from firedrake import *
  from irksome import GaussLegendre, Dt, MeshConstant, StageDerivativeNystromTimeStepper
  butcher_tableau = GaussLegendre(2)


However, we need to set up a mesh hierarchy to enable geometric multigrid
within Firedrake::

  N = 4
  nref = 3
  base = UnitSquareMesh(N, N)
  mh = MeshHierarchy(base, nref)
  msh = mh[-1]

From here, setting up the function space, manufactured solution, etc,
are just as for the regular heat equation demo::

  V = FunctionSpace(msh, "CG", 2)

  dt = Constant(10 / (N*2**nref))
  t = Constant(0.0)

  x, y = SpatialCoordinate(msh)
  uinit = sin(pi * x) * cos(pi * y)
  u = Function(V)
  u.interpolate(uinit)
  ut = Function(V)

  v = TestFunction(V)

  F = inner(Dt(u, 2), v)*dx + inner(grad(u), grad(v))*dx
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
                  "pc_python_type": "firedrake.ASMStarPC"},
              "mg_coarse": {
                  "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps"}
              }
 
These solver parameters work just fine in the :class:`.TimeStepper`::

  stepper = StageDerivativeNystromTimeStepper(
      F, butcher_tableau, t, dt, u, ut, bcs=bc,
      solver_parameters=mgparams)

The system energy is an important quantity for the wave equation, and it
should be conserved with our choice of time-stepping method::

  E = 0.5 * (inner(ut, ut) * dx + inner(grad(u), grad(u)) * dx)
  print(f"Initial energy: {assemble(E)}")

And we can advance the solution in time in typical fashion::

  while (float(t) < 1.0):
      if (float(t) + float(dt) > 1.0):
          dt.assign(1.0 - float(t))
      stepper.advance()
      t.assign(float(t) + float(dt))
      print(f"Time: {float(t)}, Energy: {assemble(E)}")


