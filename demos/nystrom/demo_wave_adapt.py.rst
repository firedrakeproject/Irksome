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
----------------------------

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

Both adaptive solves opt in to Firedrake's supermesh projection transfer
with ``mg_transfer_inject_type="project"``. Cross-mesh L2 projection satisfies

.. math::

   (u_{\rm new},v) = (u_{\rm old},v)
   \qquad\text{for every } v\in V_{\rm new}.

The right-hand side is integrated on the supermesh of the old and new meshes.
Taking :math:`v=1` preserves mass to the projection solver tolerance, including
when coarsening loses spatial detail. This transfer applies to both wave
state coefficients. It does not imply
that wave energy is conserved through mesh adaptation.

The one-stage Gauss-Legendre Nystrom method preserves these linear integral
relations on each mesh. The demo checks displacement mass after every step
and reports its absolute error every five steps.

A gradient-jump feature indicator
---------------------------------

The gradient of the CG1 displacement is constant within each triangle and
jumps across its edges. The indicator of :cite:`Kelly:1983` measures these
jumps,

.. math::

   \eta_K^2 = \frac{1}{2}\sum_{F\subset\partial K} h_F
              \int_F [\![\nabla u_h\cdot n]\!]^2\,ds,

where :math:`h_F` is the mean diameter of the two triangles that share the
edge :math:`F`. The factor :math:`\frac{1}{2}` splits the contribution of an
edge between its two triangles. We assemble :math:`\eta_K^2` into DG0 by
testing the edge integrals with the DG0 test function. The jumps vanish
wherever :math:`u_h` is linear across an edge, and they are largest where
the gradient changes most, including the crest of the pulse, where the
gradient itself vanishes. For a smooth solution, the jump is of order
:math:`h|\nabla^2 u|`, so :math:`\eta_K` is of order :math:`h_K^2`.

We use the *maximum strategy* described in Section 7.1 of :cite:`Nochetto:2009`
with a fixed reference :math:`\eta_{\rm ref}`. The callback returns these DG0
markers:

* ``1`` when :math:`\eta_K\geq0.5\,\eta_{\rm ref}` and :math:`h_K\geq2^{-4}`;
* ``-1`` when :math:`\eta_K<\theta\,\eta_{\rm ref}`;
* ``0`` otherwise.

The gap between the thresholds provides hysteresis: halving the diameter of
a cell reduces its indicator by a factor of four. The numerical thresholds
are demo choices. The reference is the largest indicator on the initial
mesh. A moving reference, such as the current maximum, would be set by the
finest cells, which the size limit keeps from refining further.

Refinement and coarsening are not symmetric. PETSc refines a marked
triangle, and then the triangles next to it that it needs to keep the mesh
conforming. In this demo, each cell marked for refinement adds about six
cells. PETSc only coarsens a group of triangles that were refined from the
same parent when all of them are marked for coarsening, so most marks for
coarsening do not remove any cell. With a fixed :math:`\theta`, the mesh
keeps growing behind the pulse. We therefore raise :math:`\theta` as the
number of cells :math:`N` exceeds a budget of :math:`N_{\rm b}=600` cells,

.. math::

   \theta = \min\left(\max\left(0.1\,(N/N_{\rm b})^4,\, 0.02\right),\, 0.25\right).

This is a feature indicator, not a reliable a posteriori error estimator for
the wave equation. It controls neither temporal error nor a prescribed
solution error.
The elliptic convergence results in the cited survey do not establish
convergence for this wave indicator.

Running the demo
----------------

The initial displacement is the Ritz projection of the Gaussian: it solves
:math:`(\nabla u_h, \nabla v) = (\nabla u_0, \nabla v)` for every :math:`v`.
The right-hand side uses the expression of :math:`u_0` in terms of
``SpatialCoordinate``, so it is evaluated exactly on each adapted mesh. On
the periodic square, this problem
determines :math:`u_h` only up to a constant. The solver therefore receives
the constant nullspace, and the demo then adds the constant that gives
:math:`u_h` the same mass as :math:`u_0`. The projection adapts the mesh with
the same markers, using the current maximum as the reference and
:math:`\theta=0.1`, so that the time steps start from a mesh that the
markers would not change much. Each time step uses one round of
refinement and coarsening, marking from the predicted displacement at the
end of that step.
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
  from firedrake.pyplot import triplot, tripcolor
  from irksome import Dt, GaussLegendre, StageDerivativeNystromTimeStepper


  def gaussian_expression(mesh):
      x, y = SpatialCoordinate(mesh)
      return (
          1.5 * exp(-180 * (sin(pi * (x - 0.32)) / pi)**2
                    - 8 * (sin(pi * (y - 0.5)) / pi)**2)
      )


  def kelly_indicator(displacement):
      mesh = displacement.function_space().mesh().unique()
      Q = FunctionSpace(mesh, "DG", 0)
      q = TestFunction(Q)
      h = CellDiameter(mesh)
      n = FacetNormal(mesh)
      squares = assemble(avg(q) * avg(h) * jump(grad(displacement), n)**2 * dS)
      return Function(Q).interpolate(sqrt(squares.riesz_representation("l2")))


  def max_value_of(f):
      with f.dat.vec_ro as values:
          return values.max()[1]


  def kelly_markers(displacement, reference=None, coarsen_fraction=0.1):
      indicator = kelly_indicator(displacement)
      Q = indicator.function_space()
      if reference is None:
          reference = max_value_of(indicator)
      if reference == 0:
          return Function(Q)
      cell_diameter = CellDiameter(Q.mesh())
      refine = conditional(
          ge(indicator, 0.5 * reference),
          conditional(ge(cell_diameter, 2**-4), 1, 0),
          0,
      )
      coarsen = conditional(lt(indicator, coarsen_fraction * reference), -1, 0)
      return Function(Q).interpolate(conditional(gt(refine, 0), 1, coarsen))


  def mark_initial(ctx, displacement):
      return kelly_markers(displacement)


  def num_cells(mesh):
      return FunctionSpace(mesh, "DG", 0).dim()


  def save_frame(displacement, time, frames):
      figure, axes = plt.subplots(figsize=(6, 5), constrained_layout=True)
      colors = tripcolor(
          displacement, axes=axes, cmap="RdBu_r", vmin=-1.5, vmax=1.5,
      )
      mesh = displacement.function_space().mesh().unique()
      triplot(
          mesh, axes=axes,
          interior_kw={"edgecolors": "black", "linewidths": 0.35},
          boundary_kw={"colors": "black", "linewidths": 0.35},
      )
      axes.set_aspect("equal")
      axes.set_axis_off()
      axes.set_title(f"t = {time:.3f}; {num_cells(mesh)} cells")
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
  u = Function(V).interpolate(gaussian_expression(mesh))
  v = TestFunction(V)
  a = inner(grad(TrialFunction(V)), grad(v)) * dx
  L = inner(grad(gaussian_expression(mesh)), grad(v)) * dx
  projection_problem = LinearVariationalProblem(a, L, u)
  projection_solver = LinearVariationalSolver(
      projection_problem,
      nullspace=VectorSpaceBasis(constant=True, comm=mesh.comm),
      solver_parameters={
          "mat_type": "aij",
          "snes_adapt_sequence": 16,
          "mg_transfer_inject_type": "project",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps",
          "mat_mumps_icntl_24": 1,
      },
      marking_callback=mark_initial,
  )
  print("Solving the adaptive initial projection")
  projection_solver.solve()
  u = projection_solver.get_solution()
  mesh = u.function_space().mesh().unique()
  u += assemble((gaussian_expression(mesh) - u) * dx) / assemble(1 * dx(domain=mesh))
  print(f"Initial projection cells: {num_cells(mesh)}")
  save_frame(u, 0.0, frames)

  reference = max_value_of(kelly_indicator(u))
  budget = 600


  def mark_within_budget(ctx, displacement, velocity):
      cells = num_cells(displacement.function_space().mesh().unique())
      coarsen_fraction = min(max(0.1 * (cells / budget)**4, 0.02), 0.25)
      return kelly_markers(displacement, reference, coarsen_fraction)


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
      "mg_transfer_inject_type": "project",
      "ksp_type": "preonly",
      "pc_type": "lu",
  }
  stepper = StageDerivativeNystromTimeStepper(
      F, GaussLegendre(1), t, dt, u, ut,
      solver_parameters=solver_parameters,
      marking_callback=mark_within_budget,
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
          cells = num_cells(stepper.u0.function_space().mesh().unique())
          print(f"t = {float(t):.3f}; {cells} cells; mass error = {mass_error:.2e}")

  output = Path(__file__).with_suffix(".gif")
  frames[0].save(
      output, save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=True
  )
  print(f"Saved adaptation movie to {output}")

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
