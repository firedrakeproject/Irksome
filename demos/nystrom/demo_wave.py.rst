Solving the Heat Equation with Irksome
======================================

Let's start with the simple wave equation on :math:`\Omega = [0,1]
\times [0,1]`, with boundary :math:`\Gamma`:

.. math::

   u_{tt} - \Delta u &= f

   u & = 0 \quad \textrm{on}\ \Gamma

for some known function :math:`f`.  At each time :math:`t`, the solution
to this equation will be some function :math:`u\in V`, for a suitable function
space :math:`V`.

We transform this into weak form by multiplying by an arbitrary test function
:math:`v\in V` and integrating over :math:`\Omega`.  We know have the
variational problem of finding :math:`u:[0,T]\rightarrow V` such
that

.. math::

   (u_t, v) + (\nabla u, \nabla v) = (f, v)

This demo implements an example used by Solin with a particular choice
of :math:`f` given below

As usual, we need to import firedrake::

  from firedrake import *

We will also need to import certain items from irksome::

  from irksome import GaussLegendre, Dt, MeshConstant, StageDerivativeNystromTimeStepper

And we will need a little bit of UFL to support using the method of
manufactured solutions::

  from ufl.algorithms.ad import expand_derivatives

We will create the Butcher tableau for a Gauss-Legendre
Runge-Kutta method, which generalizes the implicit midpoint rule::

  ns = 2
  butcher_tableau = GaussLegendre(ns)

Now we define the mesh and piecewise linear approximating space in
standard Firedrake fashion::

  N = 10

  msh = UnitSquareMesh(N, N)
  V = FunctionSpace(msh, "CG", 1)

We define variables to store the time step and current time value::

  dt = Constant(2.0 / N)
  t = Constant(0.0)

We define the initial condition for the fully discrete problem, which
will get overwritten at each time step.  For a second-order problem,
we need an initial condition for the solution and its time derivative
(in this case, taken to be zero)::

  x, y = SpatialCoordinate(msh)
  uinit = sin(pi * x) * cos(pi * y)
  u = Function(V)
  u.interpolate(uinit)
  ut = Function(V)

Now, we will define the semidiscrete variational problem using
standard UFL notation, augmented by the ``Dt`` operator from Irksome::

  v = TestFunction(V)
  F = inner(Dt(u, 2), v)*dx + inner(grad(u), grad(v))*dx
  bc = DirichletBC(V, 0, "on_boundary")

Let's use simple solver options::

  luparams = {"mat_type": "aij",
              "ksp_type": "preonly",
              "pc_type": "lu"}

Most of Irksome's magic happens in the :class:`.TimeStepper`.  It
transforms our semidiscrete form `F` into a fully discrete form for
the stage unknowns and sets up a variational problem to solve for the
stages at each time step.::

  stepper = StageDerivativeNystromTimeStepper(
      F, butcher_tableau, t, dt, u, ut, bcs=bc,
      solver_parameters=luparams)

The system energy is an important quantity for the wave equation, and it
should be conserved with our choice of time-stepping method::

  E = 0.5 * (inner(ut, ut) * dx + inner(grad(u), grad(u)) * dx)
  print(f"Initial energy: {assemble(E)}")
  
Then, we can loop over time steps, much like with 1st order systems::

  tf = 1
  while (float(t) < 1.0):
      if (float(t) + float(dt) > 1.0):
          dt.assign(1.0 - float(t))
      stepper.advance()
      print(f"Time: {float(t)}, Energy: {assemble(E)}")
      t.assign(float(t) + float(dt))

