Adaptive mesh refinement for the wave equation
==============================================

This demo solves the two-dimensional wave equation with a Nystrom stepper
and adapts the mesh to follow a moving pulse. Firedrake's nonlinear solver
adaptation uses a gradient-based marking callback: cells with a large
solution gradient are refined, while cells with a small gradient may be
coarsened. The initial Gaussian is projected onto the mesh adaptively too,
so the computation starts with a fine mesh only where the pulse is located.

The movie below shows the displacement and triangular mesh throughout the
calculation. The title reports the current number of degrees of freedom.

.. image:: demo_wave_adapt.gif
   :alt: A right-moving wave pulse with adaptive mesh refinement and coarsening.
   :align: center
   :width: 600 px

We use a smooth Gaussian pulse, localized near the left side of the unit
square. Its initial velocity is chosen as the negative horizontal derivative
of the pulse, which makes it travel to the right. A small initial mesh makes
the refinement process visible in the movie.

The callback returns one marker per cell: ``1`` requests refinement, ``-1``
requests coarsening, and ``0`` leaves a cell unchanged. The gradient is
measured in a piecewise-constant space. We refine cells above 35% of the
maximum gradient, provided their diameter is at least :math:`2^{-5}`, and
coarsen cells below 34%. The small gap between thresholds avoids rapid
refinement and coarsening around one cutoff.

The initial projection adapts fully, and the time stepper continues adapting
until halfway through the run. It then advances on the mesh produced so far,
keeping the example short while showing the pulse moving away from its
refined region.

::

  from pathlib import Path
  from io import BytesIO

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from PIL import Image

  from firedrake import *
  from firedrake.pyplot import tripcolor, triplot
  from irksome import Dt, GaussLegendre, StageDerivativeNystromTimeStepper


  def mark_by_gradient(ctx, displacement):
      mesh = displacement.function_space().mesh().unique()
      Q = FunctionSpace(mesh, "DG", 0)
      stepper = getattr(ctx, "appctx", {}).get("stepper")
      if stepper is not None and float(stepper.t) >= end_time / 2:
          return Function(Q)

      gradient = Function(Q).interpolate(sqrt(inner(grad(displacement), grad(displacement))))
      with gradient.dat.vec_ro as values:
          _, maximum = values.max()

      cell_diameter = CellDiameter(mesh)
      refine = conditional(
          gt(gradient, 0.35 * maximum),
          conditional(ge(cell_diameter, 2**-5), 1, 0),
          0,
      )
      coarsen = conditional(lt(gradient, 0.34 * maximum), -1, 0)
      return Function(Q).interpolate(conditional(gt(refine, 0), 1, coarsen))


  def save_frame(displacement, time, frames):
      figure, axes = plt.subplots(figsize=(6, 5), constrained_layout=True)
      colors = tripcolor(displacement, axes=axes, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
      triplot(
          displacement.function_space().mesh(), axes=axes,
          interior_kw={"color": "black", "linewidth": 0.25, "alpha": 0.45},
          boundary_kw={"color": "black", "linewidth": 0.4},
      )
      axes.set_aspect("equal")
      axes.set_axis_off()
      axes.set_title(f"t = {time:.3f}; {displacement.function_space().dim()} DoFs")
      figure.colorbar(colors, ax=axes, shrink=0.75)
      image = BytesIO()
      figure.savefig(image, format="png", dpi=100)
      image.seek(0)
      frames.append(Image.open(image).convert("RGB"))
      plt.close(figure)


  N = 6
  end_time = 0.3
  dt_value = 0.03
  mesh = UnitSquareMesh(N, N)
  x, y = SpatialCoordinate(mesh)
  gaussian = 25 * x * (1 - x) * y * (1 - y) * exp(
      -180 * (x - 0.32)**2 - 8 * (y - 0.5)**2
  )
  source = Function(FunctionSpace(mesh, "CG", 4)).interpolate(gaussian)

To initialize the solution, we interpolate the Gaussian into a higher-order
space, then project it onto a piecewise-linear space with the same homogeneous
boundary condition used by the wave solve. This variational solve uses
``snes_adapt_sequence=4`` to enable mesh adaptation. We save the resulting
initial condition as the first movie frame.

::

  V = FunctionSpace(mesh, "CG", 1)
  u = Function(V)
  u.interpolate(source)
  projection_test = TestFunction(V)
  projection_trial = TrialFunction(V)
  projection_form = inner(u - source, projection_test) * dx
  bc = DirichletBC(V, 0, "on_boundary")
  projection_problem = NonlinearVariationalProblem(
      projection_form,
      u,
      bcs=bc,
      J=inner(projection_trial, projection_test) * dx,
  )
  projection_solver = NonlinearVariationalSolver(
      projection_problem,
      solver_parameters={
          "mat_type": "aij",
          "snes_adapt_sequence": 4,
          "ksp_type": "preonly",
          "pc_type": "lu",
      },
      marking_callback=mark_by_gradient,
  )
  print("Solving the adaptive initial projection")
  projection_solver.solve()
  u = projection_solver.get_solution()
  print(f"Initial projection DoFs: {u.function_space().dim()}")

  frames = []
  save_frame(u, 0.0, frames)

The semidiscrete weak form is

.. math::

   (u_{tt}, v) + (\nabla u, \nabla v) = 0.

We use the one-stage Gauss-Legendre Nystrom method and the same gradient
marking callback during the adaptive portion. With zero Dirichlet boundary
conditions, the wave propagates across the square while the mesh follows its
sharp features.

::

  mesh = u.function_space().mesh().unique()
  V = u.function_space()
  x, y = SpatialCoordinate(mesh)
  gaussian = 25 * x * (1 - x) * y * (1 - y) * exp(
      -180 * (x - 0.32)**2 - 8 * (y - 0.5)**2
  )
  ut = Function(V).interpolate(-gaussian.dx(0))
  t = Constant(0.0)
  dt = Constant(dt_value)

  v = TestFunction(V)
  F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx
  stepper = StageDerivativeNystromTimeStepper(
      F, GaussLegendre(1), t, dt, u, ut, bcs=bc,
      solver_parameters={
          "mat_type": "aij",
          "snes_adapt_sequence": 1,
          "ksp_type": "preonly",
          "pc_type": "lu",
      },
      marking_callback=mark_by_gradient,
  )

At each time step, we advance the solution and record a plot of the new
displacement and mesh. The frames are combined into the GIF displayed above.
The output path is based on this script's filename, so running the demo
regenerates ``demo_wave_adapt.gif`` beside it.

::

  step_number = 0
  while float(t) < end_time - 1e-12:
      stepper.advance()
      t.assign(float(t) + float(dt))
      u = stepper.u0
      save_frame(u, float(t), frames)

      step_number += 1
      if step_number % 5 == 0:
          print(f"t = {float(t):.3f}; {u.function_space().dim()} DoFs")

  output = Path(__file__).with_suffix(".gif")
  frames[0].save(
      output, save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=True
  )
  print(f"Saved adaptation movie to {output}")
