Spectral deferred corrections for the Allen-Cahn equation
=========================================================

This demo shows how to run spectral deferred corrections (SDC) in Irksome
by composing PETSc nonlinear solvers, and compares SDC with two Newton
solvers for the same collocation problem. The test problem is the
Allen-Cahn example of Section 4.1 of :cite:`Speck:2019`.

The problem
-----------

We solve

.. math::

   u_t = \Delta u + \frac{1}{\epsilon^2} u (1 - u^2)

on :math:`\Omega = [-0.5, 0.5]^2` with periodic boundary conditions and
:math:`\epsilon = 0.04`. The initial condition

.. math::

   u_0(x) = \tanh\left(\frac{R_0 - |x|}{\sqrt{2}\epsilon}\right)

is a circle of radius :math:`R_0 = 0.25`, which shrinks over time. In the
sharp interface limit, its radius is :math:`r(t) = \sqrt{R_0^2 - 2t}`, so
the circle vanishes at :math:`t = 0.032`. As in the paper, we take 32 steps
of size :math:`\Delta t = 10^{-3}` on a :math:`128 \times 128` grid, and
use the three-stage RadauIIA method, which is the collocation method
that SDC uses on :math:`M = 3` right Radau points. The paper uses finite
differences; we use bilinear finite elements on the periodic unit square, with the
coordinates shifted to :math:`[-0.5, 0.5]^2`::

  import time
  from firedrake import *  # noqa: F403
  from irksome import Dt, MeshConstant, RadauIIA, RanaLDScheme, TimeStepper

  N = 128
  msh = PeriodicUnitSquareMesh(N, N, quadrilateral=True)
  V = FunctionSpace(msh, "CG", 1)
  x, y = SpatialCoordinate(msh)
  x = x - 0.5
  y = y - 0.5

  eps = Constant(0.04)
  R0 = 0.25
  u0 = tanh((R0 - sqrt(x**2 + y**2)) / (sqrt(2) * eps))

  MC = MeshConstant(msh)
  t = MC.Constant(0.0)
  dt = MC.Constant(1e-3)
  num_steps = 32

  butcher_tableau = RadauIIA(3)

  u = Function(V)
  v = TestFunction(V)
  F = (inner(Dt(u), v) + inner(grad(u), grad(v))
       - (1 / eps**2) * inner(u - u**3, v)) * dx

The collocation problem
-----------------------

Write the semidiscrete problem as :math:`M u_t + K(u) = 0`, where :math:`M`
is the mass matrix and :math:`K` collects the diffusion and reaction terms.
With ``stage_type="value"``, Irksome solves for the stage values
:math:`U = (U_1, \dots, U_s)`.
The stage residual is

.. math::

   F(U) = (I \otimes M)(U - \mathbb{1} \otimes u_n)
          + \Delta t\, (A \otimes I)\, K(U) = 0,

where :math:`A` is the Butcher matrix. For a collocation method,
:math:`A` is the integration matrix :math:`Q` of SDC, so :math:`F(U) = 0`
is the collocation problem that SDC solves. All three solvers below
solve this same nonlinear system, and differ only in how they do it.

We write a function that runs the simulation with a given set of solver
parameters, and optionally a monitor for the outer SNES. It returns the
final solution, the mean numbers of nonlinear and linear iterations per
step of the outer solver, and the mean wall time per step. We leave the
first step out of the timing, because it includes the generation of the
code for the forms::

  def run(solver_parameters, monitor=None, **kwargs):
      u.interpolate(u0)
      t.assign(0)
      stepper = TimeStepper(F, butcher_tableau, t, dt, u,
                            stage_type="value",
                            solver_parameters=solver_parameters, **kwargs)
      if monitor is not None:
          stepper.solver.snes.setMonitor(monitor)
      for step in range(num_steps):
          if step == 1:
              start = time.perf_counter()
          stepper.advance()
          t.assign(t + dt)
      seconds = (time.perf_counter() - start) / (num_steps - 1)
      steps, nonlinear_its, linear_its = stepper.solver_stats()
      return {"solution": u.copy(deepcopy=True),
              "nonlinear": nonlinear_its / steps,
              "linear": linear_its / steps,
              "seconds": seconds}

Baseline 1: monolithic Newton
-----------------------------

The first baseline applies Newton's method to the coupled system for all
stages, and solves each Newton step with a sparse LU factorization of the
full :math:`3 \times 3` block Jacobian::

  lu = {
      "ksp_type": "preonly",
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps",
  }

  newton = {
      "snes_type": "newtonls",
      "snes_rtol": 1e-8,
      "snes_atol": 1e-10,
      "mat_type": "aij",
      **lu,
  }

  results = {"Newton": run(newton)}

Baseline 2: Newton-Krylov with a Rana preconditioner
----------------------------------------------------

The coupled Jacobian is expensive to factor. The second baseline keeps
Newton's method on the coupled system, but solves each Newton step with
GMRES, preconditioned by the Jacobian of the residual with :math:`A`
replaced by a lower triangular matrix :math:`\tilde{A}`. Following
:cite:`Rana:2021`, we take the
:math:`LD` factors of :math:`A = LDU` (see also the :doc:`Rana demo
<demo_heat_rana.py>`).
We give the triangular tableau to the stepper as ``scheme_Jp``. The
preconditioning matrix is then block lower triangular, so a multiplicative
``fieldsplit`` with an LU factorization of each diagonal block applies its
inverse exactly by forward substitution over the stages::

  rana = {
      "snes_type": "newtonls",
      "snes_rtol": 1e-8,
      "snes_atol": 1e-10,
      "mat_type": "aij",
      "ksp_type": "gmres",
      "ksp_rtol": 1e-10,
      "pc_type": "fieldsplit",
      "pc_fieldsplit_type": "multiplicative",
      "fieldsplit": lu,
  }

  results["Newton-Krylov (RanaLD)"] = run(
      rana, scheme_Jp=RanaLDScheme(butcher_tableau))

SDC as a nonlinear Richardson iteration
---------------------------------------

SDC with the "LU trick" of :cite:`Weiser:2015` uses the same triangular
matrix, but at the nonlinear level. The LU trick factors
:math:`Q^T = L_Q U_Q` and approximates :math:`Q` by the lower triangular
matrix :math:`\tilde{A} = U_Q^T`. This is the :math:`LD` factor of
:math:`A = LDU`, which is what :func:`~irksome.RanaLDScheme` computes. One SDC sweep computes
:math:`U^{k+1}` from :math:`U^k` by solving

.. math::

   (I \otimes M)(U^{k+1} - \mathbb{1} \otimes u_n)
   + \Delta t\, (\tilde{A} \otimes I)\, K(U^{k+1})
   = -\Delta t\, ((A - \tilde{A}) \otimes I)\, K(U^k).

Let :math:`G` be the stage residual :math:`F` with :math:`A` replaced by
:math:`\tilde{A}`. The residual is linear in the Butcher matrix, so the
sweep is

.. math::

   G(U^{k+1}) = G(U^k) - F(U^k).

This is the nonlinear Richardson iteration that
:class:`~irksome.IRKAuxiliaryOperatorSNES` implements. It builds the
auxiliary residual :math:`G` through the stepper from the auxiliary
tableau in ``appctx["Fp_scheme"]``, and solves for :math:`U^{k+1}` with an
inner SNES whose options have the ``aux_`` prefix. A solution of
:math:`F(U) = 0` is a fixed point of the iteration, so SDC converges to the
same collocation solution as the two Newton solvers.

It remains to solve the auxiliary problem. Since :math:`\tilde{A}` is
lower triangular, the equation for stage :math:`i` involves only the stages
:math:`j \le i`:

.. math::

   M(U_i^{k+1} - u_n) + \Delta t \sum_{j \le i} \tilde{a}_{ij} K(U_j^{k+1})
   = \text{(frozen right-hand side)}_i.

We solve these equations one stage after another, each with the newer
values of the previous stages. That is a single forward sweep of nonlinear
block Gauss-Seidel over the stages, which PETSc provides as
``-snes_type fieldsplit`` with ``-snes_fieldsplit_type multiplicative``.
The stage space is a mixed space with one subspace per stage, and by
default there is one split per subspace, so we do not need to list the
fields. Because the system is triangular, one sweep solves it exactly. We
therefore set ``snes_max_it`` to 1 and skip the convergence test of the
inner SNES. The right-hand side :math:`G(U^k) - F(U^k)` stays fixed during
the sweep. Each stage is a nonlinear problem that looks like a backward Euler
step of size :math:`\tilde{a}_{ii}\Delta t`, which we solve with the same
Newton parameters as before, through the ``fieldsplit_`` prefix. The outer
SNES stops when the residual of the collocation problem has decreased by
:math:`10^{-8}`, as in the paper::

  sdc = {
      "snes_type": "python",
      "snes_python_type": "irksome.IRKAuxiliaryOperatorSNES",
      "snes_rtol": 1e-8,
      "snes_atol": 1e-10,
      "snes_max_it": 50,
      "aux": {
          "snes_type": "fieldsplit",
          "snes_fieldsplit_type": "multiplicative",
          "snes_max_it": 1,
          "snes_convergence_test": "skip",
          "fieldsplit": newton,
      },
  }

For SDC, the outer iterations are the sweeps. The outer SNES never solves
a linear system, so :meth:`~irksome.TimeStepper.solver_stats` reports only
the sweeps. The Newton and linear iterations happen in the stage solvers,
the children of the inner ``fieldsplit`` SNES. Its parent does not add up
their counts, so we count them with a monitor on the outer SNES. PETSc calls
the monitor after each sweep, when the stage solvers still hold the
iteration counts of that sweep::

  def run_sdc(solver_parameters):
      stage_its = {"newton": 0, "linear": 0}

      def count_stage_iterations(snes, its, fnorm):
          if its == 0:
              return
          sweep = snes.getPythonContext().solver.snes
          for stage in sweep.getFieldSplitSubSNES():
              stage_its["newton"] += stage.getIterationNumber()
              stage_its["linear"] += stage.getLinearSolveIterations()

      result = run(solver_parameters, monitor=count_stage_iterations,
                   appctx={"Fp_scheme": RanaLDScheme(butcher_tableau)})
      return {**result,
              "sweeps": result["nonlinear"],
              "nonlinear": stage_its["newton"] / num_steps,
              "linear": stage_its["linear"] / num_steps}


  results["SDC"] = run_sdc(sdc)

Practical SDC: lagging the stage Jacobians
------------------------------------------

As set up above, every Newton iteration of every stage solve assembles and
factors a new stage Jacobian, so SDC factors many more matrices than the
Newton solvers, only smaller ones. But the Jacobian of stage :math:`i`,

.. math::

   M + \Delta t\, \tilde{a}_{ii}\, K'(U_i),

changes little from one sweep to the next. We can reuse each factorization
over several iterations of the stage solver. With ``snes_lag_jacobian``
and ``snes_lag_preconditioner`` set to 10, the stage solver rebuilds and
refactors its Jacobian only every 10 iterations. The ``_persists`` options
count these iterations across solves, so a factorization is kept over
sweeps and time steps. The stage solvers persist across time steps too, so
each stage refactors about once per time step. The iterations in between
are chord iterations: they converge more slowly than Newton, but each one
is only a forward and backward substitution::

  lagged = {
      **newton,
      "snes_lag_jacobian": 10,
      "snes_lag_jacobian_persists": True,
      "snes_lag_preconditioner": 10,
      "snes_lag_preconditioner_persists": True,
  }

  sdc_lagged = {**sdc, "aux": {**sdc["aux"], "fieldsplit": lagged}}
  results["SDC (lagged Jacobian)"] = run_sdc(sdc_lagged)

Results
-------

We print the results as a table, with the difference between each
solution and the Newton solution::

  u_newton = results["Newton"]["solution"]
  print(f"{'Solver':<24}{'Sweeps':>8}{'Newton':>8}{'Linear':>8}"
        f"{'Time [s]':>10}{'Difference':>12}")
  for name, result in results.items():
      sweeps = f"{result['sweeps']:.2f}" if "sweeps" in result else "-"
      difference = errornorm(u_newton, result["solution"])
      print(f"{name:<24}{sweeps:>8}{result['nonlinear']:>8.2f}"
            f"{result['linear']:>8.2f}{result['seconds']:>10.2f}"
            f"{difference:>12.2e}")

On one core, the results are:

.. list-table::
   :header-rows: 1

   * - Solver
     - Sweeps per step
     - Newton iterations per step
     - Linear iterations per step
     - Time per step [s]
     - Difference from Newton
   * - Newton
     - \-
     - 3.06
     - 3.06
     - 0.94
     - 0
   * - Newton-Krylov (RanaLD)
     - \-
     - 3.06
     - 24.53
     - 0.82
     - 2.1e-16
   * - SDC
     - 6.75
     - 26.91
     - 26.91
     - 1.15
     - 4.4e-09
   * - SDC (lagged Jacobian)
     - 6.75
     - 33.50
     - 33.50
     - 0.57
     - 4.4e-09

The iteration counts are averages over the 32 time steps. For SDC, they
add up the iterations of all three stage solvers over all sweeps of a
step. All four solvers find the same collocation solution, to within the
tolerance of the nonlinear solvers. The two Newton solvers take the same
Newton steps, because GMRES solves each step to a tight tolerance.

The linear iterations are not equally expensive. A Newton iteration of
the monolithic solver factors and solves the coupled system for all three
stages. A GMRES iteration of the Newton-Krylov solver solves once with each
of the three diagonal blocks, which are factored once per Newton
iteration. An SDC iteration solves only one stage, with a matrix the size
of a backward Euler step. Plain SDC factors this matrix at every
iteration, about 27 times per step, and is the slowest solver here. With
lagged Jacobians, SDC factors each stage about once per step, and becomes
the fastest. The Newton solvers can lag their factorizations with the same
options, which makes them faster too, so a fair comparison should tune
all solvers. Your timings will vary with the machine.

Like the inexact variants in the paper, you can solve each stage less
accurately. For a single Newton step per stage and sweep, replace the
``fieldsplit`` options with

.. code-block:: python

  {**newton, "snes_max_it": 1, "snes_convergence_test": "skip"}

SDC then converges to the same collocation solution, with slightly more
sweeps but fewer Newton iterations in each sweep.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
