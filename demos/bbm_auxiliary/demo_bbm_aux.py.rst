Halmiltonian-structure-preserving implementation of the Benjamin-Bona-Mahoney equation
======================================================================================

This demo solves the Benjamin-Bona-Mahony equation:

.. math::

   u_t + u_x + u u_x - u_{txx} = 0

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
the :math:`H^1`-Riesz representative of the Fréchet derviative of the
Hamiltonian :math:`\frac{\delta I_3}{\delta u}` 
as the auxiliary variable :math:`\tilde{wH}`.

Standard Gauss-Legendre and continuous Petrov-Galerkin (cPG) methods conserve
the first two invariants exactly (up to roundoff and solver tolerances.  They
do quite well, but are inexact for the cubic one.  Here, we consider the
reformulation in Boris Andrews' thesis that in fact preserves the third one at
the expense of the second.  This method has an auxiliary variable in the system
and requires a continuously differentiable spatial discretization (1D Hermite
elements in this case).  The time discretization puts the main unknown in a
continuous space and the auxiliary variable in a discontinuous one.  See
equation (7.17) of Andrews' thesis for the particular formulation.


Firedrake and Irksome imports::

  from firedrake import (Constant, Function, FunctionSpace,
      PeriodicIntervalMesh, SpatialCoordinate, TestFunction, TrialFunction,
      assemble, derivative, dx, errornorm, exp, grad, inner,
      interpolate, norm, project, replace, solve, split,
  )

  from irksome import Dt, GalerkinTimeStepper, TimeQuadratureLabel

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

We need a consistent initial condition for :math:`\tilde{wH}`. ::

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

Output the initial condition to disk::

  xs = msh.coordinates.dat.data

  with open("bbm_aux_init.csv", "w") as outfile:
      outfile.write("x,u\n")
      for xcur, ucur in zip(xs, uwHtilde.subfunctions[0].dat.data[::2]):
          outfile.write("%f,%f\n" % (xcur, ucur))

Create time quadrature labels::
  
  time_order_low = 2 * (time_deg - 1)
  time_order_high = 3 * time_deg - 1

  Llow = TimeQuadratureLabel(time_order_low)
  Lhigh = TimeQuadratureLabel(time_order_high)

This tags several of the terms with a low-order time integration scheme,
but forces a higher-order method on the nonlinear term::

  u, wHtilde = split(uwHtilde)
  v, vH = split(TestFunction(Z))

  lhs = h1inner(Dt(u) + wHtilde.dx(0), v) * dx + h1inner(wHtilde, vH) * dx
  rhs = replace(Finit, {uinit: u})

  F = Llow(lhs) - Lhigh(rhs(vH))


This sets up the cPG time stepper.  There are two fields in the unknown, we
indicate the second one is an auxiliary and hence to be discretized in the DG
test space instead by passing the `aux_indices` keyword::
            
  stepper = GalerkinTimeStepper(
      F, time_deg, t, dt, uwHtilde,
      aux_indices=[1])

UFL expressions for the invariants, which we are going to track as we go
through time steps::
  
  functionals = (I1(u), I2(u), I3(u))
  invariants = [tuple(map(assemble, functionals))]
  I1ex, I2ex, I3ex = invariants[0]

  tfinal = 18.0

Do the time-stepping::

  with open("bbm_aux_invariants.csv", "w") as outfile:
      outfile.write("t,I1,I2,I3,relI1,relI2,relI3\n")
      outfile.write("%f,%f,%f,%f,%e,%e,%e\n" % (float(t), *invariants[0],
                                                0, 0, 0))
      while (float(t) < tfinal):
          if float(t) + float(dt) > tfinal:
              dt.assign(tfinal - float(t))
          stepper.advance()

          invariants.append(tuple(map(assemble, functionals)))

          i1, i2, i3 = invariants[-1]
          t.assign(float(t) + float(dt))

          print(f'{float(t):.15f}, {i1:.15f}, {i2:.15f}, {i3:.15f}')
         
          outfile.write("%f,%f,%f,%f,%e,%e,%e\n"
                        % (float(t), i1, i2, i3,
                           1-i1/I1ex, 1-i2/I2ex, 1-i3/I3ex))

  print(errornorm(uexact, uwHtilde.subfunctions[0]) / norm(uexact))

Dump out the solution at the final time step::

  with open("bbm_aux_final.csv", "w") as outfile:
      uex_final = Function(V)
      v, w = a.arguments()
      solve(a == h1inner(uexact, v) * dx, uex_final)
      outfile.write("x,uex,u,err\n")
      for xcur, uexcur, ucur in zip(xs, uex_final.dat.data[::2], uwHtilde.subfunctions[0].dat.data[::2]):
          outfile.write("%f,%f,%f,%e\n" % (xcur, uexcur, ucur, uexcur-ucur))
