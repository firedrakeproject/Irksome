Solving the Benjamin-Bona-Mahony equation with Galerkin-in-Time
===============================================================

This demo solves the Benjamin-Bona-Mahony equation:

.. math::

   u_t - u_{txx} + u_x + u u_x = 0

typically posed on :math:`\mathbb{R}` or a bounded interval with periodic
boundaries.

It is interesting because it is nonlinear (:math:`u u_x`) and also a Sobolev-type equation, with spatial derivatives acting on a time derivative.  We can obtain a weak form in the standard way:

.. math::

   (u_t, v) + (u_{tx}, v_x) + (u_x, v) + (u u_x, v) = 0

BBM is known to have a Hamiltonian structure, and there are three canonical polynomial invariants:

.. math::

   I_1 & = \int u \, \mathrm{d}x

   I_2 & = \int u^2 + (u_x)^2 \, \mathrm{d}x

   I_3 & = \int \frac{u^2}{2} + \frac{u^3}{6} \, \mathrm{d}x

We are mainly interested in accuracy and in conserving these quantities reasonably well.


Firedrake imports::

  from firedrake import *
  from irksome import Dt, TimeStepper, ContinuousPetrovGalerkinScheme

This function seems to be left out of UFL, but solitary wave solutions for BBM need it::

  def sech(x):
      return 2 / (exp(x) + exp(-x))

Set up problem parameters, etc::

  N = 1000
  L = 100
  h = L / N
  msh = PeriodicIntervalMesh(N, L)

  t = Constant(0)
  dt = Constant(10*h)

  x, = SpatialCoordinate(msh)

Here is the true solution, which is right-moving solitary wave (but not a soliton)::

  c = Constant(0.5)

  center = 30.0
  delta = -c * center

  uexact = 3 * c**2 / (1-c**2) * sech(0.5 * (c * x - c * t / (1 - c ** 2) + delta))**2

Create the approximating space and project true solution::

  V = FunctionSpace(msh, "CG", 1)
  u = project(uexact, V)
  v = TestFunction(V)

  F = (inner(Dt(u), v) * dx
       + inner((Dt(u)).dx(0), v.dx(0)) * dx
       + inner(u.dx(0), v) * dx
       + inner(u * u.dx(0), v) * dx)

For a 1d problem, we don't worry much about efficient solvers.::

  params = {"mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu"}

The Galerkin-in-Time approach should have symplectic properties::

  scheme = ContinuousPetrovGalerkinScheme(2)
  stepper = TimeStepper(F, scheme, t, dt, u, solver_parameters=params)

UFL for the mathematical invariants and containers to track them over time::

  I1 = u * dx
  I2 = (u**2 + (u.dx(0))**2) * dx
  I3 = (u**2 / 2 + u**3 / 6) * dx
  functionals = (I1, I2, I3)
  invariants = [tuple(map(assemble, functionals))]

Time-stepping loop, keeping track of :math:`I` values::

  tfinal = 18.0
  while (float(t) < tfinal):
      if float(t) + float(dt) > tfinal:
          dt.assign(tfinal - float(t))
      stepper.advance()

      invariants.append(tuple(map(assemble, functionals)))

      print('%.15f %.15f %.15f %.15f' % (float(t), *invariants[-1]))
      t.assign(float(t) + float(dt))

  print(errornorm(uexact, u) / norm(uexact))
