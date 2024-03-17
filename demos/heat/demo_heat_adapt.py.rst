Solving the Heat Equation with Irksome
======================================

Let's start with the simple heat equation on :math:`\Omega = [0,10]
\times [0,10]`, with boundary :math:`\Gamma`:

.. math::

   u_t - \Delta u &= f

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

  from irksome import RadauIIA, Dt, MeshConstant, TimeStepper

And we will need a little bit of UFL to support using the method of
manufactured solutions::

  from ufl.algorithms.ad import expand_derivatives

We will create the Butcher tableau for the 3-stage RadauIIA
Runge-Kutta method, which has an embedded scheme we can use
for adaptivity::

  butcher_tableau = RadauIIA(3)
  ns = butcher_tableau.num_stages

Now we define the mesh and piecewise linear approximating space in
standard Firedrake fashion::

  N = 100
  x0 = 0.0
  x1 = 10.0
  y0 = 0.0
  y1 = 10.0

  msh = RectangleMesh(N, N, x1, y1)
  V = FunctionSpace(msh, "CG", 1)

We define variables to store the time step, current time value, and final time::

  MC = MeshConstant(msh)
  dt = MC.Constant(10.0 / N)
  t = MC.Constant(0.0)
  Tf = MC.Constant(1.0)

This defines the right-hand side using the method of manufactured solutions::

  x, y = SpatialCoordinate(msh)
  S = Constant(2.0)
  C = Constant(1000.0)
  B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
  R = (x * x + y * y) ** 0.5
  uexact = B * atan(t)*(pi / 2.0 - atan(S * (R - t)))
  rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

We define the initial condition for the fully discrete problem, which
will get overwritten at each time step::

  u = Function(V)
  u.interpolate(uexact)

Now, we will define the semidiscrete variational problem using
standard UFL notation, augmented by the ``Dt`` operator from Irksome::

  v = TestFunction(V)
  F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx
  bc = DirichletBC(V, 0, "on_boundary")

Later demos will show how to use Firedrake's sophisticated interface
to PETSc for efficient block solvers, but for now, we will solve the
system with a direct method (Note: the matrix type needs to be
explicitly set to aij if sparse direct factorization were used with a
multi-stage method, as we wind up with a mixed problem that Firedrake
will assemble into a PETSc MatNest otherwise)::

  luparams = {"mat_type": "aij",
              "ksp_type": "preonly",
              "pc_type": "lu"}

Most of Irksome's magic happens in the :class:`.TimeStepper`.  It
transforms our semidiscrete form `F` into a fully discrete form for
the stage unknowns and sets up a variational problem to solve for the
stages at each time step.::

  stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                        stage_type="adapt", solver_parameters=luparams,
			tol=1e-3, KI=1/15, KP=0.13)

This logic is pretty self-explanatory.  We use the
:class:`.TimeStepper`'s :meth:`~.TimeStepper.advance` method, which solves the variational
problem to compute the Runge-Kutta stage values and then updates the solution.::

  while (float(t) < float(Tf)):
      if (float(t) + float(dt) > float(Tf)):
          dt.assign(float(Tf) - float(t))
      (adapt_error, adapt_dt) = stepper.advance()
      dt.assign(adapt_dt)
      print(float(t))
      t.assign(float(t) + float(dt))

Finally, we print out the relative :math:`L^2` error::

  print()
  print(norm(u-uexact)/norm(uexact))
