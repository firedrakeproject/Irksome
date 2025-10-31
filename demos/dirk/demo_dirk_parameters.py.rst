Accessing DIRK methods
=============================================================
Many practitioners favor diagonally implicit methods over fully implicit
ones since the stages can be computed sequentially rather than concurrently.
We support a range of DIRK methods and provide a convenient high-level interface
that is very similar to other RK schemes.
This demo is intended to show how to access DIRK methods seamlessly in Irksome.

This example uses the Qin Zhang symplectic DIRK to attack the mixed form of the wave
equation. Let :math:`\Omega` be the unit square with boundary :math:`\Gamma`.  We write
the wave equation as a first-order system of PDE:

.. math::

   u_t + \nabla p & = 0

   p_t + \nabla \cdot u & = 0

together with homogeneous Dirichlet boundary conditions

.. math::

   p = 0 {\quad} \textrm{on}\ \Gamma

In this form, at each time, :math:`u` is a vector-valued function in
the Sobolev space :math:`H(\mathrm{div})` and `p` is a scalar-valued
function.  If we select appropriate test functions :math:`v` and
:math:`w`, then we can arrive at the weak form (see the mixed wave
demos for more information):

.. math::

   (u_t, v) - (p, \nabla \cdot v) & = 0

   (p_t, w) + (\nabla \cdot u, w) & = 0

As in that case, we will use the next-to-lowest order Raviart-Thomas
space for :math:`u` and discontinuous piecewise linear elements for
:math:`p`.

As an example, we will use the two-stage A-stable and symplectic DIRK of Qin and
Zhang, given by Butcher tableau:

.. math::

   \begin{array}{c|cc}
   1/4 & 1/4 & 0  \\
   3/4 & 1/2 & 1/4  \\ \hline
       & 1/2 & 1/2
   \end{array}

Imports from Firedrake and Irksome::

  from firedrake import *
  from irksome import QinZhang, Dt, TimeStepper

We configure the discretization::

  N = 10
  msh = UnitSquareMesh(N, N)

  t = Constant(0.0)
  dt = Constant(1.0/N)

  V = FunctionSpace(msh, "RT", 2)
  W = FunctionSpace(msh, "DG", 1)
  Z = V*W

  v, w = TestFunctions(Z)

  butcher_tableau = QinZhang()
  
And set up the initial condition and variational problem::

  x, y = SpatialCoordinate(msh)
  up0 = project(as_vector([0, 0, sin(pi*x)*sin(pi*y)]), Z)

  u0, p0 = split(up0)

  F = inner(Dt(u0), v)*dx + inner(div(u0), w) * dx + inner(Dt(p0), w)*dx - inner(p0, div(v)) * dx

We will keep track of the energy to determine whether we're
sufficiently accurate in solving the linear system::

  E = 0.5 * (inner(u0, u0)*dx + inner(p0, p0)*dx)


When stepping with a DIRK, we only solve for one stage at a time.  Although we could configure
PETSc to try some iterative solver, here we will just use a direct method::

  params = {"mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu"}


Now, we just set the stage type to be "dirk" in Irksome and we're ready to advance in time::
  
  stepper = TimeStepper(F, butcher_tableau, t, dt, up0,
                        stage_type="dirk",
                        solver_parameters=params)

  print("Time    Energy")
  print("==============")
  while (float(t) < 1.0):
      if float(t) + float(dt) > 1.0:
          dt.assign(1.0 - float(t))

      stepper.advance()
      print("{0:1.1e} {1:5e}".format(float(t), assemble(E)))

      t.assign(float(t) + float(dt))


If all is right in the universe, you should see that the energy
remains constant.


