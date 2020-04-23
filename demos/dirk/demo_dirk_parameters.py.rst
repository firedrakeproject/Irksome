Using solver options to make gain efficiency for DIRK methods
=============================================================
Many practitioners favor diagonally implicit methods over fully implicit
ones since the stages can be computed sequentially rather than concurrently.
This demo is intended to show how Firedrake/PETSc solver options can be
used to retain this efficiency.

Abstractly, if one has a 2-stage DIRK, one has a linear system of the form

.. math::

   \left[ \begin{array}{cc} A_{11} & 0 \\ A_{12} & A_{22} \end{array} \right]
   \left[ \begin{array}{c} k_1 \\ k_2 \end{array} \right]
   &= \left[ \begin{array}{c} f_1 \\ f_2 \end{array} \right]
   
for the two stages.  This is block-lower triangular.  Traditionally, one uses
forward substitution -- solving for :math:`k_1 = A_{11}^{-1} f_1` and plugging
in to the second equation to solve for :math:`k_2`.  This can, of course,
be continued for block lower triangular systems with any number of blocks.

This can be imitated in PETSc by using a block lower triangular preconditioner
via FieldSplit.  In this case, if the diagonal blocks are inverted accurately,
one obtains an exact inverse so that a single application of the preconditioner
solves the linear system.  Hence, we can provide the efficiency of DIRKs within
the framework of Irksome with a special case of solver parameters.

This example uses this idea to attack the mixed form of the wave equation.
Let :math:`\Omega` be the unit square with boundary :math:`\Gamma`.  We write
the wave equation as a first-order system of PDE:

.. math::

   u_t + \nabla p & = 0

   p_t + \nabla \cdot u & = 0

together with homogeneous Dirichlet boundary conditions

.. math::

   p = 0 {\quad} \textrm{on}\ \Gamma

In this form, at each time, :math:`u` is a vector-valued function in
the Soboleve space :math:`H(\mathrm{div})` and `p` is a scalar-valued
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
  import numpy

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


We will need to set up parameters to pass into the solver.
PETSc-speak for performing a block lower-triangular preconditioner is
a "multiplicative field split".  And since we are claiming this is
exact, we set the Krylov method to "preonly"::
  
  params = {"mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "multiplicative"}

However, we have to be slightly careful here.  We have a two-stage
method on a mixed approximating space, so there are actually four bits
to the space we solve for stages in: `V * W * V * W`.  The natural block
structure the DIRK gives would be `(V * W) * (V * W)`.  So, we need to
tell PETSc to block the system this way::

  params["pc_fieldsplit_0_fields"] = "0,1"
  params["pc_fieldsplit_1_fields"] = "2,3"
  
This is the critical bit.  Any accurate solver for each diagonal piece
(itself a mixed system) would be fine, but for simplicity we will just
use a direct method on each stage::

  per_field = {"ksp_type": "preonly",
               "pc_type": "lu"}
  for i in range(butcher_tableau.num_stages):
      params["fieldsplit_%d" % i] = per_field

This finishes our solver specification, and we are ready to set up the
time stepper and advance in time::
      
  stepper = TimeStepper(F, butcher_tableau, t, dt, up0,
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


