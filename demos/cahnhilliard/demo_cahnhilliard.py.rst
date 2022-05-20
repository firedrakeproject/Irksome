Cahn-Hilliard Demo
==================

We are reprising the Cahn-Hilliard demo using :math:`C^1` Bell elements from
Kirby, Mitchell, "Code generation for generally mapped finite
elements," ACM TOMS 45(4), 2019.  But now, we use Irksome to perform
the time stepping.

The model is, with :math:`\Omega` the unit square,

.. math::

  \frac{\partial c}{\partial t} - \nabla \cdot M \left(\nabla\left(\frac{d f}{d c}
        - \lambda \nabla^{2}c\right)\right) = 0 \quad {\rm in}
        \ \Omega.

where :math:`f` is some typically non-convex function
(in our case, we take :math:`f(c) = 100c^2(1-c)^2`, and
:math:`\lambda` and :math:`M` are scalar parameters controlling
rates.  We are considering only the simple case of :math:`M` constant,
independent of :math:`c`.

We close the system with boundary conditions

.. math::

  M\left(\nabla\left(\frac{d f}{d c} - \lambda \nabla^{2}c\right)\right)
  \cdot n &= 0 \quad {\rm on} \ \Gamma,

  M \lambda \nabla c \cdot n &= 0 \quad {\rm on} \ \Gamma

For simplicity, we'll use a direct solver at each time step.

Boilerplate imports::

  from firedrake import *
  import numpy as np
  import os
  from irksome import Dt, GaussLegendre, TimeStepper

We create a directory to store some output pictures::

  if not os.path.exists("pictures"):
      os.makedirs("pictures")
  elif not os.path.isdir("pictures"):
      raise RuntimeError("Cannot create output directory, file of given name exists")

Set up the mesh and approximating space, including some refined ones
to allow visualizing our higher-order element on a :math:`P^1` space::

  N = 16
  msh = UnitSquareMesh(N, N)

  vizmesh = MeshHierarchy(msh, 2)[-1]

  V = FunctionSpace(msh, "Bell", 5)

Cahn-Hilliard parameters::

  lmbda = Constant(1.e-2)
  M = Constant(1)


  def dfdc(cc):
      return 200*(cc*(1-cc)**2-cc**2*(1-cc))

With Bell elements, the path of least resistance for strong boundary
conditions is a Nitsche-type method.  Here is the parameter::

  beta = Constant(250.0)

Set up the time variables and a seeded initial condition::

  dt = Constant(5.0e-6)
  T = Constant(5.0e-6)
  t = Constant(0.0)

  np.random.seed(42)
  c = Function(V)
  c.dat.data[::6] = 0.63 + 0.2*(0.5 - np.random.random(c.dat.data[::6].shape))

  v = TestFunction(V)

Now we define the semidiscrete variational problem::

  def lap(u):
      return div(grad(u))

  n = FacetNormal(msh)
  h = CellSize(msh)

  F = (inner(Dt(c), v) * dx +
       inner(M*grad(dfdc(c)), grad(v))*dx +
       inner(M*lmbda*lap(c), lap(v))*dx -
       inner(M*lmbda*lap(c), dot(grad(v), n))*ds -
       inner(M*lmbda*dot(grad(c), n), lap(v))*ds +
       inner(beta/h*M*lmbda*dot(grad(c), n), dot(grad(v), n))*ds)

Bell elements are fourth-order accurate in :math:`L^2`, so we'll use a
time-stepping scheme of comparable accuracy::

  butcher_tableau = GaussLegendre(2)

Because of the nonlinear problem, we'll need to set set some Newton
parameters as well as the linear solver::

  params = {'snes_monitor': None, 'snes_max_it': 100,
            'snes_linesearch_type': 'l2',
            'ksp_type': 'preonly',
            'pc_type': 'lu', 'mat_type': 'aij',
            'pc_factor_mat_solver_type': 'mumps'}

Set up the output::

  output = Function(FunctionSpace(vizmesh, "P", 1),
                    name="concentration")

  P5 = Function(FunctionSpace(msh, "P", 5))
  intp = Interpolator(c, P5)

  def interpolate_output():
      intp.interpolate()
      return prolong(P5, output)

Save the initial condition to a file::

  import matplotlib.pyplot as plt
  interpolate_output()
  cs = tripcolor(output, vmin=0, vmax=1)
  plt.colorbar(cs)
  plt.savefig('pictures/init.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

Now let's set up the time stepper::

  stepper = TimeStepper(F, butcher_tableau, t, dt, c,
                        solver_parameters=params)

And advance the solution in time::

  while float(t) < float(T):
      if (float(t) + float(dt)) >= 1.0:
          dt.assign(1.0 - float(t))
      stepper.advance()
      t.assign(float(t) + float(dt))
      print(float(t), float(dt))

We'll save a snapshout of the final state::

  interpolate_output()
  cs = tripcolor(output, vmin=0, vmax=1)
  plt.colorbar(cs)
  plt.savefig('pictures/final.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

And report the amount of overshoot we get in the method::

  print(np.max(c.dat.data[::6]))
