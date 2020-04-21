Using solver options to make gain efficiency for DIRK methods
=============================================================
Many practitioners favor diagonally implicit methods over fully implicit
ones since the stages can be computed sequentially rather than concurrently.
This demo is intended to show how Firedrake/PETSc solver options can be
used to retain this efficiency.

Abstractly, if one has a 2-stage DIRK, one has a linear system of the form

.. math:
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

This example uses this idea to attack the mixed form of the wave equation:

Last login: Sun Apr 19 21:03:59 on ttys001
c02yf10ljgh8:cahnhilliard robert_kirby$ top
c02yf10ljgh8:cahnhilliard robert_kirby$ top
c02yf10ljgh8:cahnhilliard robert_kirby$ cd ~/Code/Irksome/demos/
c02yf10ljgh8:demos robert_kirby$ ls
Makefile				demo_dirk_parameters.py			demo_lowlevel_mixed_heat.py		navier_stokes_demo.py
cahnhilliard				demo_heat_adaptivestepper.py.rst	demo_nitsche_heat.py			nse_steady_demo.py
circle.step				demo_heat_pc.py				demo_nse_unsteady.py			preconditioning
coarse-circle.geo			demo_lowlevel_homogbc.py		heat
coarse-circle.msh			demo_lowlevel_inhomogbc.py		mixed_wave
c02yf10ljgh8:demos robert_kirby$ mkdir dirk
c02yf10ljgh8:demos robert_kirby$ cd dirk/
c02yf10ljgh8:dirk robert_kirby$ ls
c02yf10ljgh8:dirk robert_kirby$ mv ../demo_dirk_parameters.py .
c02yf10ljgh8:dirk robert_kirby$ ls
demo_dirk_parameters.py
c02yf10ljgh8:dirk robert_kirby$ emacs demo_dirk_parameters.py 
c02yf10ljgh8:dirk robert_kirby$ mv demo_dirk_parameters.py demo_dirk_parameters.py.rst
c02yf10ljgh8:dirk robert_kirby$ emacs demo_dirk_parameters.py.rst &
[1] 98250
c02yf10ljgh8:dirk robert_kirby$ 2020-04-21 15:56:17.802 Emacs-x86_64-10_14[98250:744064] Failed to initialize color list unarchiver: Error Domain=NSCocoaErrorDomain Code=4864 "*** -[NSKeyedUnarchiver _initForReadingFromData:error:throwLegacyExceptions:]: non-keyed archive cannot be decoded by NSKeyedUnarchiver" UserInfo={NSDebugDescription=*** -[NSKeyedUnarchiver _initForReadingFromData:error:throwLegacyExceptions:]: non-keyed archive cannot be decoded by NSKeyedUnarchiver}

c02yf10ljgh8:dirk robert_kirby$ less ../mixed_wave/
README                               demo_RTwave.py.rst                   demo_RTwave_adaptive_stepper.py.rst  
demo_RTwave.py                       demo_RTwave_adaptive_stepper.py      
c02yf10ljgh8:dirk robert_kirby$ less ../mixed_wave/demo_RTwave.py
c02yf10ljgh8:dirk robert_kirby$ less ../mixed_wave/demo_RTwave.py.rst 





























Let :math:`Omega` be the unit square with boundary :math:`Gamma`.  We write
the wave equation as a first-order system of PDE:

.. math:

   u_t + grad(p) & = 0
   p_t + div(u) & = 0

together with homogeneous Dirichlet boundary conditions

.. math:

   p = 0 

In this form, at each time, :math:`u` is a vector-valued function in
the Soboleve space :math:`H(\mathrm{div})` and `p` is a scalar-valued
function.  If we select appropriate test functions :math:`v` and
:math:`w`, then we can arrive at the weak form (see the mixed wave
demos for more information):

.. math:

   (u_t, v) - (p, div(v)) & = 0

   (p_t, w) + (div(u), w) & = 0

As in that case, we will use the next-to-lowest order Raviart-Thomas
space for :math:`u` and discontinuous piecewise linear elements for
:math:`p`.  

As an example, we will use the two-stage A-stable and symplectic DIRK of Qin and
Zhang, given by Butcher tableau:

.. math:

   \begin{tabular}{cc|c}
   1/4 & 1/4 & 0 \\
   3/4 & 1/2 & 1/4 \\ \hline
       & 1/2 & 1/2
   \end{tabular}

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


