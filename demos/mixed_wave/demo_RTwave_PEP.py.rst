:orphan:

Solving the Mixed Wave Equation: Energy conservation
====================================================

Let :math:`\Omega` be the unit square with boundary :math:`\Gamma`.  We write
the wave equation as a first-order system of PDE:

.. math::

   u_t + \nabla p & = 0
   
   p_t + \nabla \cdot u & = 0

together with homogeneous Dirichlet boundary conditions

.. math::

   p = 0 \quad \textrm{on}\ \Gamma

In this form, at each time, :math:`u` is a vector-valued function in the Sobolev space :math:`H(\mathrm{div})` and `p` is a scalar-valued function.  If we select appropriate test functions :math:`v` and :math:`w`, then we can arrive at the weak form

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
  from irksome import Dt, MeshConstant, TimeStepper, PEPRK

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

  MC = MeshConstant(msh)
  t = MC.Constant(0.0)
  dt = MC.Constant(0.2/N)

The PEP RK methods of de Leon, Ketcheson, and Ranoch offer explicit RK methods
that preserve the energy up to a given order in step size.  They have more
stages than classical explicit methods but have much better energy conservation.::

  butcher_tableau = PEPRK(4, 2, 5)

Like the heat equation demo, we are just using a direct method to solve the
system at each time step::

  params = {"snes_type": "ksponly",
            "ksp_type": "cg",
            "pc_type": "icc"}

  stepper = TimeStepper(F, butcher_tableau, t, dt, up0,
                        stage_type="explicit",
                        solver_parameters=params)


And, as with the heat equation, our time-stepping logic is quite simple.  At each time step, we print out the energy in the system::

  print("Time    Energy")
  print("==============")

  while (float(t) < 1.0):
      if float(t) + float(dt) > 1.0:
          dt.assign(1.0 - float(t))

      stepper.advance()

      t.assign(float(t) + float(dt))
      print("{0:1.1e} {1:5e}".format(float(t), assemble(E)))

If all is well with the world, the energy will be nearly identical (up
to roundoff error) at each time step because the PEP methods conserve
energy to quite high order.  The reader can compare this to the mixed
wave demo using Gauss-Legendre methods (which exactly conserve energy up
to roundoff and solver tolerances.


