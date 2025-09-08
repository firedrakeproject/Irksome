Solving the Navier-Stokes Equations with Irksome
================================================

Let's consider the lid-driven cavity problem on :math:`\Omega = [0,1]
\times [0,1]`, with boundary :math:`\Gamma_T \cup \Gamma`, where
:math:`\Gamma_T` is the top of the domain, :math:`0 \leq x \leq 1, y=1`:

.. math::

   u_t + u \cdot \nabla u - \frac{1}{Re}\Delta u + \nabla p &= 0

   \nabla\cdot u & = 0

   u & = (0,0) \quad \textrm{on}\ \Gamma

   u & = (1,0) \quad \textrm{on}\ \Gamma_T

At each time :math:`t`, the solution to this equation will be some
functions :math:`(u,p)\in V\times W`, for a suitable function spaces
:math:`V, W`.

We transform this into weak form by multiplying arbitrary test
functions :math:`v\in V` and :math:`w\in W` and integrating over
:math:`\Omega`.  This gives the variational problem of finding
:math:`u:[0,T]\rightarrow V` and :math:`p:[0,T]\rightarrow W` such
that

.. math::

   (u_t, v) + (u \cdot \nabla u, v) + \frac{1}{Re}(\nabla u, \nabla v) - (p, \nabla\cdot v) & = 0

   (\nabla \cdot u, w) & = 0

As usual, we need to import firedrake::

  from firedrake import *

We will also need to import certain items from irksome::

  from irksome import RadauIIA, Dt, MeshConstant, TimeStepper

We will create the Butcher tableau for the two-stage RadauIIA
Runge-Kutta method::

  butcher_tableau = RadauIIA(2)
  ns = butcher_tableau.num_stages

Now we define the mesh and Taylor-Hood approximating space in
standard Firedrake fashion::

  N = 32
  msh = UnitSquareMesh(N, N)
  V = VectorFunctionSpace(msh, "CG", 2)
  W = FunctionSpace(msh, "CG", 1)
  Z = V*W

We define variables to store the time step and current time value, as
well as the Reynolds number::

  MC = MeshConstant(msh)
  dt = MC.Constant(1.0 / N)
  t = MC.Constant(0.0)
  Re = MC.Constant(10.0)


We define the solution over the product space, which will get
overwritten at each time step::

  up = Function(Z)
  u, p = split(up)

Now, we will define the semidiscrete variational problem using
standard UFL notation, augmented by the ``Dt`` operator from Irksome::

  v, w = TestFunctions(Z)
  F = (inner(Dt(u), v) * dx + inner(dot(u, grad(u)), v) * dx
       + 1/Re * inner(grad(u), grad(v)) * dx - inner(p, div(v)) * dx
       + inner(div(u), w) * dx)

  bcs = [DirichletBC(Z.sub(0), as_vector([0, 0]), (1, 2, 3)),
         DirichletBC(Z.sub(0), as_vector([1, 0]), (4,))]

Later demos will show how to use Firedrake's sophisticated interface
to PETSc for efficient solvers, but for now, we will solve the
system with a direct method (Note: the matrix type needs to be
explicitly set to aij if sparse direct factorization were used with a
multi-stage method, as we wind up with a mixed problem that Firedrake
will assemble into a PETSc MatNest otherwise)::

  luparams = {"mat_type": "aij",
              "snes_type": "newtonls",
              "snes_monitor": None,
              "ksp_type": "preonly",
              "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps",
              "snes_linesearch_type": "l2",
              "snes_force_iteration": 1,
              "snes_rtol": 1e-8,
              "snes_atol": 1e-8,
              }

Since this problem is ill-posed, we specify a nullspace vector to
remove the possible constant "shift" in the pressure::

  nsp = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=msh.comm)])


Most of Irksome's magic happens in the :class:`.TimeStepper`.  It
transforms our semidiscrete form `F` into a fully discrete form for
the stage unknowns and sets up a variational problem to solve for the
stages at each time step.::

  stepper = TimeStepper(F, butcher_tableau, t, dt, up, bcs=bcs,
                        solver_parameters=luparams, nullspace=nsp)

This logic is pretty self-explanatory.  We use the
:class:`.TimeStepper`'s :meth:`~.TimeStepper.advance` method, which
solves the variational problem to compute the Runge-Kutta stage values
and then updates the solution.::

  for _ in range(N):
      print(f"Stepping from time {float(t)}")
      stepper.advance()
      t.assign(float(t) + float(dt))

Finally, we can visualize results of the simulation using Firedrake's
plotting capabilities::

  import matplotlib.pyplot as plt
  from firedrake.pyplot import streamplot
  u_, p_ = up.subfunctions
  fig, axes = plt.subplots()
  streamplot(u_, resolution=0.02, axes=axes)
  axes.set_aspect("equal")
  fig.savefig("demo_nse_streamlines.png")
