The Wave equation with adaptive time stepping
=============================================

We now build on our previous example, present example builds on that, but uses the adaptive time-stepping
capabilities.

Let :math:`Omega` be the unit square with boundary :math:`Gamma`.  As
in the other demo, we have the weak form of the wave equation:

.. math:

   (u_t, v) - (p, div(v)) & = 0

   (p_t, w) + (div(u), w) & = 0

Here is some typical Firedrake boilerplate and the construction of a simple
mesh and the approximating spaces::

  from firedrake import *
  from irksome import GaussLegendre, Dt, AdaptiveTimeStepper

  N = 10

  msh = UnitSquareMesh(N, N)
  V = FunctionSpace(msh, "RT", 2)
  W = FunctionSpace(msh, "DG", 1)
  Z = V*W

Now we can build the initial condition, which has zero velocity and a sinusoidal displacement::

  x, y = SpatialCoordinate(msh)
  up0 = project(as_vector([0, 0, sin(pi*x)*sin(pi*y)]), Z)
  u0, p0 = split(up0)


We build the variational form in UFL::

  v, w = TestFunctions(Z)
  F = inner(Dt(u0), v)*dx + inner(div(u0), w) * dx + inner(Dt(p0), w)*dx - inner(p0, div(v)) * dx

Energy conservation is an important principle of the wave equation, and we can
test how well the spatial discretization conserves energy by creating a
UFL expression and evaluating it at each time step::

  E = 0.5 * (inner(u0, u0)*dx + inner(p0, p0)*dx)

The time and time step variables::

  t = Constant(0.0)
  dt = Constant(1.0/N)

The two-stage Gauss-Legendre method is, like all instances of that family,
A-stable and symplectic.  This gives us a fourth order method in time, although
our spatial accuracy is of lower order.  Feel free to experiment!::

  butcher_tableau = GaussLegendre(2)

Like the heat equation demo, we are just using a direct method to solve the
system at each time step::

  params = {"mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu"}

Now, we are using the :class:`AdaptiveTimeStepper` mechanism.  The
constructor is similar to the regular time stepper, but takes some
optional extra parameters specifying the temporal truncation error
tolerance and the minimal acceptable time step::

  stepper = AdaptiveTimeStepper(F, butcher_tableau, t, dt, up0,
                                tol=1.e-3, dtmin=1.e-5,
                                solver_parameters=params)

Now, the stepping logic is very similar to before, although irksome
will print out information about what is going on with time-step
selection/adaptation::

  initial_energy = assemble(E)
  while (float(t) < 1.0):
      if float(t) + float(dt) > 1.0:
          dt.assign(1.0 - float(dt))
      err = stepper.advance()
      print(float(t), float(dt), assemble(E))

      t.assign(float(t) + float(dt))
  final_energy = assemble(E)

As with fixed-size time steps, energy should be very well conserved
with the Gauss-Legendre method, although not with non-symplectic
methods.  (One disadvantage of the adaptive time-stepper is that the
system matrix must be re-factored when the time step changes, so an
effective iterative method would be of interest.)

