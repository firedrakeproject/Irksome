Preserving Structure for the Heat Equation
==========================================

Here we'll consider the unforced heat equation on :math:`\Omega =
[0,1] \times [0,1]`, with boundary :math:`\Gamma`:

.. math::

   u_t - \Delta u &= 0 \quad \textrm{in}\ \Omega

   u(x,0) & = u_0(x) \quad \textrm{in}\ \Omega

   u(x,t) & = 0 \quad \textrm{on}\ \Gamma

for some known function :math:`u_0`.

We transform this into weak form by multiplying by an arbitrary test function
:math:`v\in V` and integrating over :math:`\Omega`.  We now have the
variational problem of finding :math:`u:[0,T]\rightarrow V` such
that

.. math::

   (u_t, v) + (\nabla u, \nabla v) = 0

with :math:`u(x,0) = \Pi u_0(x)` for some projection or interpolation
operator, :math:`\Pi`.  Here, we focus on how some IRK discretizations
preserve structure, given by an energy law.  Choosing :math:`v=u` and
doing some calculus, the above variational form becomes

.. math::

   \frac{\partial}{\partial t} \frac{1}{2} (u,u) = -(\nabla u, \nabla u)

Using Crank-Nicolson as the integrator, it's possible to show a discrete form of this law holds, with

.. math::

   \frac{ (u^{n+1},u^{n+1}) - (u^n,u^n)}{2} = -(\nabla u^{n+1/2}, \nabla u^{n+1/2})

Here, we'll ensure that this happens.

As usual, we need to import firedrake::

  from firedrake import *

We will also need to import certain items from irksome::

  from irksome import GaussLegendre, Dt, TimeStepper

We will create the Butcher tableau for the lowest-order Gauss-Legendre
Runge-Kutta method, which is equivalent to Crank-Nicolson in this case::

  butcher_tableau = GaussLegendre(1)
  ns = butcher_tableau.num_stages

Now we define the mesh and piecewise linear approximating space in
standard Firedrake fashion::

  N = 100

  msh = UnitSquareMesh(N, N)
  V = FunctionSpace(msh, "CG", 1)

We define variables to store the time step and current time value::

  dt = Constant(0.005)
  t = Constant(0.0)

We use a manufactured solution that gives a zero right-hand side::

  x, y = SpatialCoordinate(msh)
  uexact = sin(pi*x)*sin(2*pi*y)*exp(-5*pi**2*t)

We define the initial condition for the fully discrete problem, which
will get overwritten at each time step::

  u = Function(V)
  u.interpolate(uexact)

Now, we will define the semidiscrete variational problem using
standard UFL notation, augmented by the ``Dt`` operator from Irksome::

  v = TestFunction(V)
  F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx
  bc = DirichletBC(V, 0, "on_boundary")

Other demos show how to use Firedrake's sophisticated interface to
PETSc for efficient block solvers, so we will solve the system with a
direct method::

  luparams = {"mat_type": "aij",
              "ksp_type": "preonly",
              "pc_type": "lu"}

We use the midpoint of each interval as a sample point::

  sample_point = [0.5]

This gets passed into Irksome's :class:`.TimeStepper` as a keyword
argument.::

  stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                        solver_parameters=luparams,
			sample_points=sample_point)

From here, we use the :class:`.TimeStepper`'s
:meth:`~.TimeStepper.advance` method to solve the variational problem
to compute the Runge-Kutta stage values and update the solution.  To
verify the structure preservation, we compute the norm-squared of the
solution at each timestep and the norm-squared of the solution
gradient at the midpoint of the timestep after each timestep,
accessible through the variable :meth:`stepper.sample_values[0]`.
From the discrete energy law above, one half of the finite
differencing of the solution norm-squared should be equal to, but
opposite in sign of the gradient norm squared at the midpoints.::

  norm_form = inner(u,u)*dx
  solution_norms = [assemble(norm_form)]
  deriv_norms = []
  deriv_form = inner(grad(stepper.sample_values[0]),grad(stepper.sample_values[0]))*dx
  
  while (float(t) < 0.2):
      if (float(t) + float(dt) > 0.2):
          dt.assign(0.2 - float(t))
      stepper.advance()
      t.assign(float(t) + float(dt))
      solution_norms.append(assemble(norm_form))
      deriv_norms.append(assemble(deriv_form))
      print(f"At time {float(t):.3f}, LHS is {(solution_norms[-1]-solution_norms[-2])/(2*float(dt)):.4e}, RHS is {deriv_norms[-1]:.4e}, difference is {(solution_norms[-1]-solution_norms[-2])/(2*float(dt)) + deriv_norms[-1]:.4e}")

