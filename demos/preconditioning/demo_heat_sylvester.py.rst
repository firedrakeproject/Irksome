Preconditioning by a Sylvester equation
=======================================

This demo solves the stage system of a fully implicit Runge-Kutta
method with :class:`.IRKSylvesterPC`, which needs only solves with a
backward Euler matrix and with the mass matrix, whatever the number of
stages.  We use the heat equation on :math:`\Omega = [0,10] \times
[0,10]`, with boundary :math:`\Gamma`, as in the other preconditioning
demos:

.. math::

   (u_t, v) + (\nabla u, \nabla v) = (f, v)

For a method with Butcher matrix :math:`A` and :math:`s` stages, the
stage system is

.. math::

   (I \otimes M + \Delta t A \otimes K) k = b,

with the mass matrix :math:`M` and the stiffness matrix :math:`K`.  We
write the stage unknowns as the columns of an :math:`n \times s` matrix
:math:`X`.  After multiplication by :math:`(\Delta t A)^{-1} \otimes I`,
the stage system is the Sylvester equation

.. math::

   K_\sigma X + M X S_\sigma^T = B, \qquad
   K_\sigma = K + \sigma M, \quad S_\sigma = (\Delta t A)^{-1} - \sigma I,

where :math:`\sigma` is the smallest real part of the eigenvalues of
:math:`(\Delta t A)^{-1}`.  :math:`K_\sigma` is a backward Euler matrix
with the time step :math:`1/\sigma`.  The preconditioner solves this
equation with the extended Krylov subspace method (EKSM), which needs
only solves with :math:`K_\sigma` and with :math:`M`.  For a linear
problem with constant coefficients the Sylvester equation is the stage
system, thus the preconditioner is a solver.  For a nonlinear problem
the preconditioner uses one Jacobian for all the stages, which is
evaluated again at each Newton iteration.

We set up the problem as in the other demos, on a mesh hierarchy so
that we can use geometric multigrid later::

  from firedrake import *
  from irksome import RadauIIA, Dt, MeshConstant, TimeStepper

  butcher_tableau = RadauIIA(3)

  N = 32
  x0 = 0.0
  x1 = 10.0
  y0 = 0.0
  y1 = 10.0

  base = RectangleMesh(N // 4, N // 4, x1, y1)
  mh = MeshHierarchy(base, 2)
  msh = mh[-1]

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
  rhs = Dt(uexact) - div(grad(uexact))

  u = Function(V)
  u.interpolate(uexact)
  v = TestFunction(V)
  F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx
  bc = DirichletBC(V, 0, "on_boundary")

The outer Krylov method is FGMRES, because the preconditioner is an
iterative solve.  The preconditioner applies the stage matrix without
assembly, so we use a matrix-free stage matrix.  The options of the
preconditioner have the prefix ``sylvester_``: ``sylvester_ksp`` for
EKSM, ``sylvester_K`` for the solves with :math:`K_\sigma` and
``sylvester_M`` for the solves with :math:`M`.  By default both inner
solves use a sparse direct solver.  EKSM stops when the residual of the
stage system is below ``sylvester_ksp_rtol``.  We set it below the
FGMRES tolerance, so that FGMRES converges in one iteration::

  params = {"snes_type": "ksponly",
            "mat_type": "matfree",
            "ksp_type": "fgmres",
            "ksp_rtol": 1e-8,
            "pc_type": "python",
            "pc_python_type": "irksome.IRKSylvesterPC",
            "sylvester_ksp_rtol": 1e-9,
            "sylvester_ksp_converged_reason": None}

  stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                        solver_parameters=params)

  for _ in range(4):
      stepper.advance()
      t.assign(float(t) + float(dt))

  steps, nonlinear_its, linear_its = stepper.solver_stats()
  print(f"FGMRES iterations per time step: {linear_its/steps:.2f}")

The solves with :math:`K_\sigma` are the expensive part on fine
meshes.  :math:`K_\sigma` is the matrix of one implicit Euler step, so
the same solvers apply.  Here we use conjugate gradients with geometric
multigrid for :math:`K_\sigma`, and conjugate gradients with Jacobi
preconditioning for :math:`M`.  The inner tolerances must be tight
enough for the EKSM tolerance.  By default the tolerance of the
:math:`K_\sigma` solves increases as EKSM converges
(``sylvester_adaptive_tol``)::

  params["sylvester_K"] = {"ksp_type": "cg",
                           "ksp_rtol": 1e-10,
                           "pc_type": "mg"}
  params["sylvester_M"] = {"ksp_type": "cg",
                           "ksp_rtol": 1e-10,
                           "pc_type": "jacobi"}

  u.interpolate(uexact)
  stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                        solver_parameters=params)

  for _ in range(4):
      stepper.advance()
      t.assign(float(t) + float(dt))

  steps, nonlinear_its, linear_its = stepper.solver_stats()
  print(f"FGMRES iterations per time step: {linear_its/steps:.2f}")

Finally, we print the relative :math:`L^2` error::

  print(norm(u-uexact)/norm(uexact))
