Solving the Mixed Wave Equation: Energy conservation 
====================================================

Let :math:`\Omega` be the unit square with boundary :math:`\Gamma`.  We write
the wave equation as a first-order system of PDE:

.. math::

   u_t + grad(p) & = 0
   
   p_t + div(u) & = 0

together with homogeneous Dirichlet boundary conditions

.. math::

   p = 0 \quad \textrm{on}\ \Gamma

In this form, at each time, :math:`u` is a vector-valued function in the Soboleve space :math:`H(\mathrm{div})` and `p` is a scalar-valued function.  If we select appropriate test functions :math:`v` and :math:`w`, then we can arrive at the weak form

.. math::

   (u_t, v) - (p, \nabla \cdot v) & = 0

   (p_t, w) + (\nabla \cdot u, w) & = 0

Note that in mixed formulations, the Dirichlet boundary condition is weakly
enforced via integration by parts rather than strongly in the definition of
the approximating space.

In this example, we will use the next-to-lowest order Raviart-Thomas elements
for the velocity variable :math:`u` and discontinuous piecewise linear
polynomials for the scalar variable :math:`p`.

Here is some typical Firedrake boilerplate and the construction of a simple
mesh and the approximating spaces::
   
  from firedrake import *
  from irksome import GaussLegendre, Dt, TimeStepper
  import numpy

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

  stepper = TimeStepper(F, butcher_tableau, t, dt, up0,
                        solver_parameters=params)


And, as with the heat equation, our time-stepping logic is quite simple.  At easch time step, we print out the energy in the system::

  print("Time    Energy")
  print("==============")

  while (float(t) < 1.0):
      if float(t) + float(dt) > 1.0:
          dt.assign(1.0 - float(t))

      stepper.advance()

      t.assign(float(t) + float(dt))
      print("{0:1.1e} {1:5e}".format(float(t), assemble(E)))

If all is well with the world, the energy will be nearly identical (up
to roundoff error) at each time step because the GL methods are
symplectic and applied to a linear Hamiltonian system.  As an
exercise, the reader should edit this code to use other RK methods.
In particular, LobattoIIIC and Backward Euler or other Radau methods
as well as the symplectic DIRK by Qin and Zhang.

