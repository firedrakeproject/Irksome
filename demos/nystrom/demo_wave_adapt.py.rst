Adaptive wave equation with a Nystrom stepper
=============================================

.. image:: demo_wave_adapt.gif
   :alt: A right-moving wave pulse with adaptive mesh refinement and coarsening.

An adaptive L2 projection computes the initial Gaussian before the movie
starts. The Nystrom stepper then adapts the right-moving wave at each step.

::

  from pathlib import Path
  from io import BytesIO

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from PIL import Image

  from firedrake import *
  from firedrake.pyplot import tripcolor, triplot
  from irksome import Dt, GaussLegendre, NystromStepper


  def gaussian_expression(mesh):
      x, y = SpatialCoordinate(mesh)
      return (
          25 * x * (1 - x) * y * (1 - y)
          * exp(-180 * (x - 0.32)**2 - 8 * (y - 0.5)**2)
      )


  def gaussian_displacement(mesh, degree=1):
      V = FunctionSpace(mesh, "CG", degree)
      displacement = Function(V)
      displacement.interpolate(gaussian_expression(mesh))
      return displacement


  def gradient_markers(displacement):
      mesh = displacement.function_space().mesh().unique()
      Q = FunctionSpace(mesh, "DG", 0)
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
      markers = Function(Q).interpolate(conditional(gt(refine, 0), 1, coarsen))
      return markers


  def mark_by_gradient(ctx, displacement):
      return gradient_markers(displacement)


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
  dt_value = 0.005
  mesh = UnitSquareMesh(N, N)
  frames = []
  V = FunctionSpace(mesh, "CG", 1)
  source = gaussian_displacement(mesh, degree=4)
  u = Function(V).interpolate(source)
  projection_test = TestFunction(V)
  projection_trial = TrialFunction(V)
  projection_form = inner(u - source, projection_test) * dx
  boundary_condition = DirichletBC(V, 0, "on_boundary")
  projection_problem = NonlinearVariationalProblem(
      projection_form,
      u,
      bcs=boundary_condition,
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
  save_frame(u, 0.0, frames)

  V = u.function_space()
  mesh = V.mesh().unique()
  ut = Function(V).interpolate(-gaussian_expression(mesh).dx(0))
  t = Constant(0.0)
  dt = Constant(dt_value)

  v = TestFunction(V)
  F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx
  bc = DirichletBC(V, 0, "on_boundary")
  solver_parameters = {
      "mat_type": "aij",
      "snes_adapt_sequence": 1,
      "ksp_type": "preonly",
      "pc_type": "lu",
  }
  stepper = NystromStepper(
      F, GaussLegendre(1), t, dt, u, ut, bcs=bc,
      solver_parameters=solver_parameters,
      marking_callback=mark_by_gradient,
  )

  step_number = 0
  while float(t) < end_time - 1e-12:
      stepper.advance()
      t.assign(float(t) + float(dt))
      save_frame(stepper.u0, float(t), frames)
      step_number += 1
      if step_number % 5 == 0:
          print(f"t = {float(t):.3f}; {stepper.u0.function_space().dim()} DoFs")

  output = Path(__file__).with_suffix(".gif")
  frames[0].save(
      output, save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=True
  )
  print(f"Saved adaptation movie to {output}")
