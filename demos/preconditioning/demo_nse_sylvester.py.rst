Navier-Stokes equations with a Sylvester-equation preconditioner
================================================================

This demo solves the unsteady incompressible Navier-Stokes equations
with a fully implicit Runge-Kutta method, and solves the stage systems
with :class:`.IRKSylvesterPC`.  See the heat equation demo
(:doc:`demo_heat_sylvester.py`) for the idea of the preconditioner.
Here we show two more features: the treatment of the pressure, which
has no time derivative, and iterative inner solvers for a saddle point
problem: monolithic multigrid with Vanka relaxation for
:math:`K_\sigma`, and a Schur complement preconditioner for the mass
matrix.

The equations are

.. math::

   u_t + u \cdot \nabla u - \nu \Delta u + \nabla p &= 0

   \nabla \cdot u &= 0

in the channel :math:`\Omega = [0, 4] \times [0, 1]`, with a parabolic
inflow profile of time-dependent strength at :math:`x = 0`, no slip on
the walls and the natural (do-nothing) condition at the outflow
:math:`x = 4`.  Thus the pressure is unique.

The semi-discrete problem is a differential-algebraic equation: the
pressure has no time derivative, and the mass matrix :math:`M` is
singular.  :class:`.IRKSylvesterPC` finds the fields without a time
derivative and changes the pressure unknowns and the divergence
equations of the stage system with the matrix :math:`I + \beta
S_\sigma`.  This change is exact.  The Sylvester equation then has the
invertible matrix

.. math::

   M_\beta = \begin{bmatrix} M_u & \beta B^T \\ \beta B & 0 \end{bmatrix}

in place of :math:`M`, where :math:`B` is the divergence matrix.  We
have :math:`M_\beta = D M_1 D` with :math:`D = \mathrm{diag}(I, \beta
I)`, so the preconditioner sets up the solver for :math:`M_1` and applies
:math:`\beta` by diagonal scaling.  Thus the solver of :math:`M_1` does
not depend on :math:`\beta` or on the time step.  The other matrix of the
Sylvester equation is the matrix of one implicit
Euler step with the time step :math:`1/\sigma`, linearized at the mean
of the stage values,

.. math::

   K_\sigma = \begin{bmatrix} \sigma M_u + \nu A + N(\bar u) & B^T \\ B & 0
   \end{bmatrix}.

We start with the usual imports::

  from firedrake import *
  from irksome import RadauIIA, Dt, MeshConstant, TimeStepper

Problem set-up
--------------

We use Taylor-Hood elements and the three-stage Radau IIA method.  We
build a mesh hierarchy, so that we can solve with :math:`K_\sigma` by
geometric multigrid::

  butcher_tableau = RadauIIA(3)

  base = RectangleMesh(8, 2, 4.0, 1.0)
  mh = MeshHierarchy(base, 2)
  msh = mh[-1]
  V = VectorFunctionSpace(msh, "CG", 2)
  W = FunctionSpace(msh, "CG", 1)
  Z = V * W

  MC = MeshConstant(msh)
  t = MC.Constant(0.0)
  dt = MC.Constant(0.05)
  nu = 0.01

  x, y = SpatialCoordinate(msh)
  up = Function(Z)
  u, p = split(up)
  v, q = TestFunctions(Z)
  F = (inner(Dt(u), v)*dx + inner(dot(grad(u), u), v)*dx
       + nu*inner(grad(u), grad(v))*dx - p*div(v)*dx - div(u)*q*dx)

  inflow = as_vector([6*y*(1 - y)*(1 + sin(2*pi*t)/2), 0])
  bcs = [DirichletBC(Z.sub(0), inflow, 1),
         DirichletBC(Z.sub(0), 0, (3, 4))]

The solver for :math:`K_\sigma`
-------------------------------

:math:`K_\sigma` is the matrix of one implicit Euler step of the
Navier-Stokes equations, thus we use a standard solver for it: FGMRES
with monolithic geometric multigrid.  The relaxation is additive Schwarz
on Vanka patches: each patch contains the velocity and pressure
unknowns in the star of a vertex, and the velocity unknowns in the
closure of the star.  On the coarsest mesh we use a sparse direct
solver::

  vanka = {"pc_type": "python",
           "pc_python_type": "firedrake.ASMVankaPC",
           "pc_vanka_construct_dim": 0,
           "pc_vanka_exclude_subspaces": 1}

  K_params = {"mat_type": "aij",
              "ksp_type": "fgmres",
              "ksp_rtol": 1e-8,
              "pc_type": "mg",
              "mg_levels": {"ksp_type": "gmres",
                            "ksp_max_it": 2,
                            "ksp_convergence_test": "skip",
                            **vanka},
              "mg_coarse": {"ksp_type": "preonly",
                            "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps"}}

The solver for :math:`M_1`
--------------------------

:math:`M_1` is a saddle point matrix.  We solve with it by FGMRES with
a block factorization.  The velocity block is a mass matrix, and we
apply Jacobi preconditioning to it.  The Schur complement :math:`-B
M_u^{-1} B^T` is spectrally equivalent to :math:`-B
\,\mathrm{diag}(M_u)^{-1} B^T`, which is a pressure Laplacian.  PETSc
assembles this matrix from the blocks of :math:`M_1`
(``pc_fieldsplit_schur_precondition: selfp``), thus it has the boundary
conditions of the discrete problem.  We apply one algebraic multigrid
cycle to it.  :math:`M_1` depends neither on the state nor on the time
step, thus this solver is set up only once::

  M_params = {"mat_type": "aij",
              "ksp_type": "fgmres",
              "ksp_rtol": 1e-8,
              "pc_type": "fieldsplit",
              "pc_fieldsplit_type": "schur",
              "pc_fieldsplit_schur_fact_type": "full",
              "pc_fieldsplit_schur_precondition": "selfp",
              "fieldsplit_0": {"ksp_type": "preonly",
                               "pc_type": "jacobi"},
              "fieldsplit_1": {"ksp_type": "preonly",
                               "pc_type": "hypre"}}

Solver parameters
-----------------

The outer solver is Newton's method with FGMRES.  For a nonlinear
problem the preconditioner is not a solver, so EKSM only needs a loose
tolerance.  We test it on the residual of the Sylvester equation,
which costs nothing (``sylvester_residual``)::

  params = {"snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_rtol": 1e-8,
            "snes_monitor": None,
            "mat_type": "matfree",
            "ksp_type": "fgmres",
            "ksp_rtol": 1e-8,
            "pc_type": "python",
            "pc_python_type": "irksome.IRKSylvesterPC",
            "sylvester_residual": "sylvester",
            "sylvester_ksp_rtol": 1e-3,
            "sylvester_K": K_params,
            "sylvester_M": M_params}

The solver of :math:`K_\sigma` is set up again at each Newton
iteration, at the new linearization state::

  stepper = TimeStepper(F, butcher_tableau, t, dt, up, bcs=bcs,
                        solver_parameters=params)

  for _ in range(4):
      stepper.advance()
      t.assign(float(t) + float(dt))

  steps, nonlinear_its, linear_its = stepper.solver_stats()
  print(f"Newton iterations per time step: {nonlinear_its/steps:.2f}")
  print(f"FGMRES iterations per Newton iteration: {linear_its/nonlinear_its:.2f}")
