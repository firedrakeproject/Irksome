Solving the Heat Equation with a Multistep Method in Irksome
======================================

Consider the heat equation on :math:`\Omega = [0,10] \times [0,10]`, with boundary :math:`\Gamma`:

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

  from irksome import Dt, MeshConstant, MultistepTimeStepper, MultistepMethod

We will use the 3-step Backward Difference Formula::

  method = MultistepMethod('BDF', 3)

Now we define the mesh and piecewise linear approximating space in
standard Firedrake fashion::

  N = 100
  x0 = 0.0
  x1 = 10.0
  y0 = 0.0
  y1 = 10.0

  msh = RectangleMesh(N, N, x1, y1)
  V = FunctionSpace(msh, "CG", 1)

We define variables to store the time step and current time value::

  MC = MeshConstant(msh)
  dt = MC.Constant(5.0 / N)
  t = MC.Constant(0.0)

This defines the right-hand side using the method of manufactured solutions::

  x, y = SpatialCoordinate(msh)
  S = Constant(2.0)
  C = Constant(1000.0)
  B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
  R = (x * x + y * y) ** 0.5
  uexact = B * atan(t)*(pi / 2.0 - atan(S * (R - t)))
  rhs = Dt(uexact) - div(grad(uexact))

We define the initial condition for the fully discrete problem, which
will get overwritten at each time step::

  u = Function(V)
  u.interpolate(uexact)

Now, we will define the semidiscrete variational problem using
standard UFL notation, augmented by the ``Dt`` operator from Irksome::

  v = TestFunction(V)
  F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx
  bc = DirichletBC(V, 0, "on_boundary")

We'll use a basic solver for this demo::

  luparams = {"mat_type": "aij",
              "ksp_type": "preonly",
              "pc_type": "lu"}

Much like Irksome's :class:`.TimeStepper`, the :class:`.MultistepTimeStepper` automates the 
transformation of our semidiscrete form `F` into a fully discrete form for
the next approximate value and sets up a variational problem to solve at each time step.


An s-step multistep method requires starting values :math:`u^0,\dots, u^{s - 1}` in order to begin the approximation. Irksome provides two ways to 
set these values. If you wish to set the starting values manually, you can create a :class:`.MultistepTimeStepper` as shown below, but with no `startup_parameters`. You can 
then access the list :class:`.MultistepTimeStepper.us` which contains the starting approximations, assign the desired starting values, and advance `t` by hand. 
On the other hand, if you wish to use a method which Irksome supports to obtain these starting values, then Irksome allows this process to be completed automatically.

We'll use the automated startup procedure. This requires defining a :class:`dict` of keyword arguments used to setup a :class:`.TimeStepper`. The :class:`.TimeStepper` is then used to compute the 
initial approximations needed to start the method. We'll import RadauIIA and use the backward Euler method to obtain our starting values. Formally, the backward Euler method is only first order accurate, 
and we wish to use it obtain the starting values for the third-order accurate BDF(3) method. A crude way to increase the accuracy of the starting values is to use a smaller timestep for the startup procedure. 
Here, we use timesteps of size :math:`\Delta t / 8` for the backward Euler method which is accessible through the keyword `dt_div`. The keyword `stepper_kwargs` allows for easy customization of the startup :class:`.TimeStepper`.::

  from irksome import RadauIIA

  startup_stepper_kwargs = {'stage_type': 'value', 
                            'solver_parameters': luparams}

  startup_parameters = {'tableau': RadauIIA(1),
                        'dt_div': 8,
                        'stepper_kwargs': startup_stepper_kwargs}

We can then set up the :class:`.MultistepTimeStepper` as follows::

  stepper = MultistepTimeStepper(F, method, t, dt, u, bcs=bc, 
                                 solver_parameters=luparams, 
                                 startup_parameters=startup_parameters)

Note that the creation of a :class:`.MultistepTimeStepper` with parameters for the startup procedure will not automatically solve for the required starting values. 
One must call the :class:`.MultistepTimeStepper`'s :meth:`~.MultistepTimeStepper.startup` method, which will internally construct a :class:`.TimeStepper`, solve the the required starting 
values, and advance `t` to `t + (s-1)*dt`:::

  stepper.startup()
  print(f'The starting values have been computed. The current time is {float(t)}')

This logic is pretty self-explanatory.  We use the
:class:`.MultistepTimeStepper`'s :meth:`~.MultistepTimeStepper.advance` method, which solves the variational
problem to compute the next approximate value and updates the solution.::

  while (float(t) < 1.0):
    if (float(t) + float(dt) > 1.0):
        dt.assign(1.0 - float(t))
    stepper.advance()
    t.assign(float(t) + float(dt))
    print(float(t))

Now, check the relative :math:`L^2` error::

  print()
  print(norm(u-uexact)/norm(uexact))
