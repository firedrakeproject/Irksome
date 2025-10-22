Invariant preserving implementation of the Benjamin-Bona-Mahoney equation
=========================================================================

This demo solves the Benjamin-Bona-Mahony equation:

.. math::

   u_t + u_x + u u_x - u_{txx} = 0

posed on a bounded interval with periodic boundaries.

BBM is known to have a Hamiltonian structure, and there are several canonical polynomial invariants:

.. math::

   I_1 & = \int u \, dx

   I_2 & = \int u^2 + (u_x)^2 \, dx

   I_3 & = \int \frac{u^2}{2} + \frac{u^3}{6} \, dx

Standard Gauss-Legendre and continuous Petrov-Galerkin (cPG) methods conserve
the first two invariants exactly (up to roundoff and solver tolerances.  They
do quite well, but are inexact for the cubic one.
Here, we consider the reformulation in Boris Andrews' thesis that in fact
preserves the third one at the expense of the second.
This method has an auxiliary variable in the system and requires a continuously differentiable spatial discretization (1d Hermite elements in this case).
The time discretization puts the main unknown in a continuous space and the
auxiliary variable in a discontinuous one.  See equation (7.17) of Andrews'
thesis for the particular formulation.


Firedrake imports::

  from firedrake import (
      Constant, Function, FunctionSpace,
      PeriodicIntervalMesh, SpatialCoordinate, TestFunction, TestFunctions,
      assemble, dx, errornorm, exp, grad, inner, interpolate, norm, project,
      solve, split
  )


  from irksome import Dt, GalerkinTimeStepper
  from irksome.labeling import TimeQuadratureLabel

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

  uwHtilde = Function(Z)
  uwHtilde.subfunctions[0].project(uexact)

  wHtilde = Function(V)

We need a consistent initial condition for :math:`\tilde{wH}`::
  
  v = TestFunction(V)
  Finit = (inner(wHtilde, v) * dx
           + inner(wHtilde.dx(0), v.dx(0)) * dx
           - inner(uexact + 0.5 * uexact**2, v) * dx)
  solve(Finit==0, wHtilde)
  uwHtilde.subfunctions[1].assign(wHtilde)

  u, wHtilde = split(uwHtilde)

Output the initial condition to disk::

  xs = msh.coordinates.dat.data[::2]

  with open("bbm_aux_init.csv", "w") as outfile:
      outfile.write("x,u\n")
      for xcur, ucur in zip(xs, uwHtilde.subfunctions[0].dat.data):
          outfile.write("%f,%f\n" % (xcur, ucur))

  v, vH = TestFunctions(Z)

Create temporal quadrature rules in FIAT::
  
  time_order_low = 2 * (time_deg - 1)
  time_order_high = 3 * time_deg - 1

  Llow = TimeQuadratureLabel(time_order_low)
  Lhigh = TimeQuadratureLabel(time_order_high)


  def h1inner(u, v):
      return inner(u, v) + inner(grad(u), grad(v))


This tags several of the terms with a low-order time integration scheme,
but forces a higher-order method on the nonlinear term::

  F = Llow(h1inner(Dt(u), v) * dx
         - 0.5 * h1inner(wHtilde, v.dx(0)) * dx
         + 0.5 * h1inner(wHtilde.dx(0), v) * dx
         + h1inner(wHtilde, vH) * dx) \
         - Lhigh(inner(u + 0.5 * u**2, vH) * dx)


This sets up the cPG time stepper.  There are two fields in the unknown, we indicate the second one is an auxiliary and hence to be discretized in the DG
space instead by passing the `aux_indices` keyword::
            
  stepper = GalerkinTimeStepper(
      F, time_deg, t, dt, uwHtilde,
      aux_indices=[1])

UFL expressions for the invariants, which we are going to track as we go
through time steps::
  
  I1 = u * dx
  I2 = (u**2 + (u.dx(0))**2) * dx
  I3 = (u**2 / 2 + u**3 / 6) * dx

  I1s = []
  I2s = []
  I3s = []

  tfinal = 18.0

Do the time-stepping::

  with open("bbm_aux_invariants.csv", "w") as outfile:
      outfile.write("t,I1,I2,I3,relI1,relI2,relI3\n")
      outfile.write("%f,%f,%f,%f,%e,%e,%e\n" % (float(t), assemble(I1),
                                                assemble(I2), assemble(I3),
                                                0, 0, 0))
      while (float(t) < tfinal):
          if float(t) + float(dt) > tfinal:
              dt.assign(tfinal - float(t))
          stepper.advance()

          I1s.append(assemble(I1))
          I2s.append(assemble(I2))
          I3s.append(assemble(I3))

          i1 = I1s[-1]
          i2 = I2s[-1]
          i3 = I3s[-1]
          t.assign(float(t) + float(dt))

          print(
              f'{float(t):.15f}, {i1:.15f}, {i2:.15f}, {i3:.15f}')
         
          outfile.write("%f,%f,%f,%f,%e,%e,%e\n"
                        % (float(t),
                           I1s[-1], I2s[-1], I3s[-1],
                           1-I1s[-1]/I1s[0],
                           1-I2s[-1]/I2s[0],
                           1-I3s[-1]/I3s[0]))

  print(errornorm(uexact, uwHtilde.subfunctions[0]) / norm(uexact))

Dump out the solution at the final time step::

  with open("bbm_aux_final.csv", "w") as outfile:
      uex_final = project(uexact, V)
      outfile.write("x,uex,u,err\n")
      for xcur, uexcur, ucur in zip(xs, uex_final.dat.data, uwHtilde.subfunctions[0].dat.data):
          outfile.write("%f,%f,%f,%e\n" % (xcur, uexcur, ucur, uexcur-ucur))
