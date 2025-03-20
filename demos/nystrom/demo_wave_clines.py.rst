Solving the wave equation with monolithic multigrid
===================================================

This reprise of the wave equation demo uses a monolithic multigrid
algorithm to perform time advancement.

We consider the telegraph equation on :math:`\Omega = [0,1]
\times [0,1]`, with boundary :math:`\Gamma`: giving rise to the weak form

.. math::

   (u_{tt}, v) + (u_t, v) + (\nabla u, \nabla v) = 0

We perform similar imports and setup as before::

  from firedrake import *
  from irksome import GaussLegendre, Dt, MeshConstant, StageDerivativeNystromTimeStepper
  butcher_tableau = GaussLegendre(2)


We're going to use a stage-segregated preconditioner, and we have access to AMG
for the fields.  No need for a mesh hierarchy::

  N = 32
  msh = UnitSquareMesh(N, N)

From here, setting up the function space, manufactured solution, etc,
are just as for the regular wave equation demo::

  V = FunctionSpace(msh, "CG", 2)

  dt = Constant(8 / N)
  t = Constant(0.0)

  x, y = SpatialCoordinate(msh)
  uinit = sin(pi * x) * cos(pi * y)
  u = Function(V)
  u.interpolate(uinit)
  ut = Function(V)

  v = TestFunction(V)

  F = inner(Dt(u, 2), v)*dx + inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx
  bc = DirichletBC(V, 0, "on_boundary")

And now for the solver parameters.  Note that we are solving a
block-wise system with all stages coupled together.  We will segregate
those stages with the preconditioner from Clines/Howle/Long, and use
hypre on the diagonal blocks::

  params = {"mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "gmres",
            "ksp_monitor": None,
            "pc_type": "python",
            "pc_python_type": "irksome.ClinesLD",
            "aux": {
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "multiplicative",
		"fieldsplit": {
		  "ksp_type": "preonly",
		  "pc_type": "lu"
		}
            }}
 
These solver parameters work just fine in the stepper.::

  stepper = StageDerivativeNystromTimeStepper(
      F, butcher_tableau, t, dt, u, ut, bcs=bc,
      solver_parameters=params)

The system energy is an important quantity for the wave equation.  It is also
important for the telegraph method, but should decay exponentially over time
instead of being conserved.::

  E = 0.5 * (inner(ut, ut) * dx + inner(grad(u), grad(u)) * dx)
  print(f"Initial energy: {assemble(E)}")

And we can advance the solution in time in typical fashion::

  while (float(t) < 1.0):
      if (float(t) + float(dt) > 1.0):
          dt.assign(1.0 - float(t))
      stepper.advance()
      t.assign(float(t) + float(dt))
      print(f"Time: {float(t)}, Energy: {assemble(E)}")


