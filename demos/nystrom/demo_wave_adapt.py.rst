Adaptive mesh refinement for the periodic wave equation
========================================================

This demo solves the two-dimensional wave equation with a Nystrom stepper
on a doubly periodic unit square. The mesh follows a localized pulse through
refinement and coarsening. Supermesh projection preserves the integrals of
the displacement and velocity when the mesh changes.

.. image:: demo_wave_adapt.gif
   :alt: A wave pulse with adaptive mesh refinement and coarsening.
   :align: center
   :width: 600 px

Conservation and periodicity
---------------------------

The weak form is

.. math::

   (u_{tt}, v) + (\nabla u, \nabla v) = 0.

The periodic space contains the constant test function. Setting :math:`v=1`
shows that :math:`\int_\Omega u_t\,dx` is constant. Thus the displacement
mass :math:`\int_\Omega u\,dx` is constant if the initial velocity has zero
mean. We choose a smooth periodic analogue of a Gaussian and obtain its
initial velocity by projecting :math:`-\partial_x u_h` into the periodic
space. This derivative has zero integral. It gives the pulse an initial
rightward bias; a localized two-dimensional wave is not an exact translating
one-dimensional profile.

Both adaptive solves opt in to Firedrake's coefficient transfer option
``snes_adapt_transfer="project"``. It requires a Firedrake version that
supports this option. Cross-mesh L2 projection satisfies

.. math::

   (u_{\rm new},v) = (u_{\rm old},v)
   \qquad\text{for every } v\in V_{\rm new}.

The right-hand side is integrated on the supermesh of the old and new meshes.
Taking :math:`v=1` preserves mass to the projection solver tolerance, including
when coarsening loses spatial detail. This transfer applies to the initial
projection's source and to both wave state coefficients. It does not imply
that wave energy is conserved through mesh adaptation.

The one-stage Gauss-Legendre Nystrom method preserves these linear integral
relations on each mesh. The demo checks displacement mass after every step
and reports its absolute error every five steps.

A gradient-based feature indicator
----------------------------------

For each triangle :math:`K`, let :math:`h_K` be its diameter. We use

.. math::

   \eta_K = h_K^{1+d/2}|\nabla u_h|_K
          = h_K^2|\nabla u_h|_K, \qquad d=2.

This scaling follows the heuristic first-order interpolation argument in the
`deal.II step-9 tutorial <https://www.dealii.org/current/doxygen/deal.II/step_9.html>`_.
Here the CG1 gradient is constant within each triangle, so we evaluate it
directly in DG0. The cell-size factor makes the indicator decrease as a
smooth region is refined.

We use the *maximum strategy* described in Section 7.1 of Nochetto, Siebert
and Veeser, `Theory of adaptive finite element methods: An introduction
<https://doi.org/10.1007/978-3-642-03413-8_12>`_ (2009). With
:math:`\eta_{\max}=\max_K\eta_K`, the callback returns these DG0 markers:

* ``1`` when :math:`\eta_K\geq0.5\eta_{\max}` and :math:`h_K\geq2^{-5}`;
* ``-1`` when :math:`\eta_K<0.1\eta_{\max}`;
* ``0`` otherwise, or everywhere when all indicators vanish.

The numerical thresholds and coarsening rule are demo choices. Their gap
provides hysteresis: halving the diameter reduces the indicator by a factor
of four if the gradient stays fixed. PETSc also enforces mesh conformity and
requires compatible coarsening requests.

This is a feature indicator, not a reliable a posteriori error estimator for
the wave equation. It controls neither temporal error nor a prescribed
solution error, and displacement alone can miss features carried by velocity.
The elliptic convergence results in the cited survey do not establish
convergence for this wave indicator.

Running the demo
----------------

The initial L2 projection uses four adaptation rounds. Each time step uses
one round, marking from the predicted displacement at the end of that step.
The output path is based on the script's filename, so running the demo
regenerates ``demo_wave_adapt.gif`` beside it.

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


  def gaussian_expression(mesh):
      x, y = SpatialCoordinate(mesh)
      return (
          1.5 * exp(-180 * (sin(pi * (x - 0.32)) / pi)**2
                    - 8 * (sin(pi * (y - 0.5)) / pi)**2)
      )


  def gaussian_displacement(mesh, degree=1):
      V = FunctionSpace(mesh, "CG", degree)
      displacement = Function(V)
      displacement.interpolate(gaussian_expression(mesh))
      return displacement


  def gradient_markers(displacement):
      mesh = displacement.function_space().mesh().unique()
      Q = FunctionSpace(mesh, "DG", 0)
      cell_diameter = CellDiameter(mesh)
      indicator = Function(Q).interpolate(
          cell_diameter**2 * sqrt(inner(grad(displacement), grad(displacement)))
      )
      with indicator.dat.vec_ro as values:
          _, maximum = values.max()
      if maximum == 0:
          return Function(Q)

      refine = conditional(
          ge(indicator, 0.5 * maximum),
          conditional(ge(cell_diameter, 2**-5), 1, 0),
          0,
      )
      coarsen = conditional(lt(indicator, 0.1 * maximum), -1, 0)
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
  mesh = PeriodicUnitSquareMesh(N, N)
  frames = []
  V = FunctionSpace(mesh, "CG", 1)
  source = gaussian_displacement(mesh, degree=4)
  u = Function(V).interpolate(source)
  projection_test = TestFunction(V)
  projection_trial = TrialFunction(V)
  projection_form = inner(u - source, projection_test) * dx
  projection_problem = NonlinearVariationalProblem(
      projection_form,
      u,
      J=inner(projection_trial, projection_test) * dx,
  )
  projection_solver = NonlinearVariationalSolver(
      projection_problem,
      solver_parameters={
          "mat_type": "aij",
          "snes_adapt_sequence": 4,
          "snes_adapt_transfer": "project",
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
  ut = Function(V).project(-u.dx(0), solver_parameters={"ksp_rtol": 1e-12})
  initial_mass = assemble(u * dx)
  t = Constant(0.0)
  dt = Constant(dt_value)

  v = TestFunction(V)
  F = inner(Dt(u, 2), v) * dx + inner(grad(u), grad(v)) * dx
  solver_parameters = {
      "mat_type": "aij",
      "snes_adapt_sequence": 1,
      "snes_adapt_transfer": "project",
      "ksp_type": "preonly",
      "pc_type": "lu",
  }
  stepper = StageDerivativeNystromTimeStepper(
      F, GaussLegendre(1), t, dt, u, ut,
      solver_parameters=solver_parameters,
      marking_callback=mark_by_gradient,
  )

  step_number = 0
  while float(t) < end_time - 1e-12:
      stepper.advance()
      t.assign(float(t) + float(dt))
      mass_error = abs(assemble(stepper.u0 * dx) - initial_mass)
      assert mass_error < 1e-10 * max(1.0, abs(initial_mass))
      save_frame(stepper.u0, float(t), frames)
      step_number += 1
      if step_number % 5 == 0:
          print(f"t = {float(t):.3f}; {stepper.u0.function_space().dim()} DoFs; "
                f"mass error = {mass_error:.2e}")

  output = Path(__file__).with_suffix(".gif")
  frames[0].save(
      output, save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=True
  )
  print(f"Saved adaptation movie to {output}")
