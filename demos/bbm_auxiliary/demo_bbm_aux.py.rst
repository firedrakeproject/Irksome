Halmiltonian-structure-preserving implementation of the Benjamin-Bona-Mahoney equation
======================================================================================

This demo solves the Benjamin-Bona-Mahony equation:

.. math::

   u_t - u_{txx} + u_x + u u_x = 0

posed on a bounded interval with periodic boundaries.

BBM is known to have a Hamiltonian structure, and there are several canonical polynomial invariants:

.. math::

   I_1 & = \int u \, dx

   I_2 & = \int u^2 + (u_x)^2 \, dx

   I_3 & = \int \frac{u^2}{2} + \frac{u^3}{6} \, dx

The BBM invariants are the total momentum :math:`I_1`, the :math:`H^1`-energy
norm :math:`I_2`, and the Hamiltonian :math:`I_3`.  
The Hamiltonian formulation reads

.. math::

   \partial_t (u - u_{xx}) & = - \partial_x \frac{\delta I_3}{\delta u}

The numerical scheme in this demo introduces
the :math:`H^1`-Riesz representative of the Fréchet derivative of the
Hamiltonian :math:`\frac{\delta I_3}{\delta u}` 
as the auxiliary variable :math:`\tilde{wH}`.

Standard Gauss-Legendre and continuous Petrov-Galerkin (cPG) methods conserve
the first two invariants exactly (up to roundoff and solver tolerances).  They
do quite well, but are inexact for the cubic one.  Here, we consider the
reformulation in Andrews and Farrell, "Enforcing conservation laws and dissipation
inequalities numerically via auxiliary variables" (arXiv:2407.11904, to appear
in SIAM J. Scientific Computing) that preserves the third invariant at
the expense of the second.  This method has an auxiliary variable in the system
and requires a continuously differentiable spatial discretization (1D Hermite
elements in this case).  The time discretization puts the main unknown in a
continuous space and the auxiliary variable in a discontinuous one.  See
equation (7.17) of Boris Andrews' thesis for the particular formulation.


Firedrake, Irksome, and other imports::

  from firedrake import (Constant, Function, FunctionSpace,
      PeriodicIntervalMesh, SpatialCoordinate, TestFunction, TrialFunction,
      assemble, derivative, dx, errornorm, exp, grad, inner,
      interpolate, norm, plot, project, replace, solve, split
  )

  from irksome import Dt, GalerkinTimeStepper, TimeQuadratureLabel

  import matplotlib.pyplot as plt
  import numpy


  def sech(x):
      return 2 / (exp(x) + exp(-x))


  N = 8000
  L = 100
  h = L / N
  msh = PeriodicIntervalMesh(N, L)

  c = Constant(0.5)

  t = Constant(0)
  dt = Constant(10*h)

  x, = SpatialCoordinate(msh)

  center = 40.0
  delta = -c * center

  uexact = 3 * c**2 / (1-c**2) \
      * sech(0.5 * (c * x - c * t / (1 - c ** 2) + delta))**2

  space_deg = 3
  time_deg = 1

This sets up the function space for the unknown :math:`u` and
auxiliary variable :math:`\tilde{wH}`::

  V = FunctionSpace(msh, "Hermite", space_deg)
  Z = V * V

We project the initial condition on :math:`u`, but we also need a consistent initial condition for the auxiliary variable. 

Let :math:`F = \frac{\delta I_3}{\delta u}` be the Fréchet derivative of the
Hamiltonian. We need to find :math:`\tilde{wH}` such that :math:`(\tilde{wH}, v)_{H^1} = F(v)`.
::

  def h1inner(u, v):
      return inner(u, v) + inner(grad(u), grad(v))

  def I1(u):
      return u * dx

  def I2(u):
      return h1inner(u, u) * dx

  def I3(u):
      return (u**2 / 2 + u**3 / 6) * dx

  uwHtilde = Function(Z)
  uinit, wHinit = uwHtilde.subfunctions
  
  v = TestFunction(V)
  w = TrialFunction(V)
  a = h1inner(w, v) * dx
  Finit = derivative(I3(uinit), uinit, v)

  solve(a == h1inner(uexact, v)*dx, uinit)
  solve(a == Finit, wHinit)

Visualize the initial condition::

  fig, axes = plt.subplots(1)
  plot(Function(FunctionSpace(msh, "CG", 1)).interpolate(uinit), axes=axes)
  axes.set_title("Initial condition")
  plt.savefig("bbm_init.png")
  
Create time quadrature labels::
  
  time_order_low = 2 * (time_deg - 1)
  time_order_high = 3 * time_deg - 1

  Llow = TimeQuadratureLabel(time_order_low)
  Lhigh = TimeQuadratureLabel(time_order_high)

This tags several of the terms with a low-order time integration scheme,
but forces a higher-order method on the nonlinear term::

  u, wHtilde = split(uwHtilde)
  v, vH = split(TestFunction(Z))

  Flow = h1inner(Dt(u) + wHtilde.dx(0), v) * dx + h1inner(wHtilde, vH) * dx
  Fhigh = replace(Finit, {uinit: u})

  F = Llow(Flow) - Lhigh(Fhigh(vH))


This sets up the cPG time stepper.  There are two fields in the unknown, we
indicate the second one is an auxiliary and hence to be discretized in the DG
test space instead by passing the `aux_indices` keyword::
            
  stepper = GalerkinTimeStepper(
      F, time_deg, t, dt, uwHtilde, aux_indices=[1])

UFL expressions for the invariants, which we are going to track as we go
through time steps::

  times = [float(t)]
  functionals = (I1(u), I2(u), I3(u))
  invariants = [tuple(map(assemble, functionals))]
  I1ex, I2ex, I3ex = invariants[0]

  tfinal = 18.0

Do the time-stepping::

  while (float(t) < tfinal):
      if float(t) + float(dt) > tfinal:
          dt.assign(tfinal - float(t))
      stepper.advance()

      invariants.append(tuple(map(assemble, functionals)))

      i1, i2, i3 = invariants[-1]
      t.assign(float(t) + float(dt))
      times.append(float(t))

      print(f'{float(t):.15f}, {i1:.15f}, {i2:.15f}, {i3:.15f}')

Visualize invariant preservation::

  axes.clear()
  invariants = numpy.array(invariants)

  lbls = ("I1", "I2", "I3")

  for i in (0, 1, 2):
      plt.plot(times, invariants[:, i], label=lbls[i])
  axes.set_title("Invariants over time")
  axes.legend()
  plt.savefig("invariants.png")
  axes.clear()

  for i in (0, 1, 2):
      plt.plot(times, 1.0 - invariants[:, i]/invariants[0, i], label=lbls[i])
  axes.set_title("Relative error in invariants over time")
  axes.legend()  
  plt.savefig("invariant_errors.png")

Visualize the solution at final time step::

  axes.clear()
  plot(Function(FunctionSpace(msh, "CG", 1)).interpolate(uwHtilde.subfunctions[0]), axes=axes)
  axes.set_title(f"Solution at time {tfinal}")
  plt.savefig("bbm_final.png")
  
