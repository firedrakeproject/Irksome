
Rana/Howle/et al preconditioning
====================================================

This demo applies a method suggested in:

Rana, Howle, Long, Meek, Milestone "A new block preconditioner for implicit Runge-Kutta methods for parabolic PDE problems," SISC 2021

to our ongoing heat equation demonstration problem on :math:`\Omega = [0,10]
\times [0,10]`, with boundary :math:`\Gamma`, giving rise to the weak form

.. math::

   (u_t, v) + (\nabla u, \nabla v) = (f, v)

A multi-stage RK method applied to the heat equation gives a
block-structured system.  The on-diagonal blocks are quite similar to
what one obtains from a backward Euler discretization of the equation.

With a 2-stage method, we have

.. math::
   
   \left[ \begin{array}{cc} A_{11} & A_{12} \\ A_{21} & A_{22} \end{array} \right]
   \left[ \begin{array}{c} k_1 \\ k_2 \end{array} \right]
   &= \left[ \begin{array}{c} f_1 \\ f_2 \end{array} \right]

The suggestion in the paper is to approximate the Butcher matrix A with
with a triangular approximation.  For example, if A = LDU, then one could approximate A with LD or DU.  This gives rise to a block triangular preconditioner

.. math::

  P = \left[ \begin{array}{cc} \tilde{A}_{11} & 0 \\ \tilde_{A}_{21} & \tilde{A}_{22} \end{array} \right]


This allows one to leverage an existing methodology for a low order
method like backward Euler for the diagonal blocks.  Empirical results
suggest that this method is strongly stage-independent.

Common set-up for the problem::

  from firedrake import *  # noqa: F403
  from ufl.algorithms.ad import expand_derivatives
  from irksome import LobattoIIIC, TimeStepper, Dt, MeshConstant

  butcher_tableau = LobattoIIIC(3)

  N = 16

  x0 = 0.0
  x1 = 10.0
  y0 = 0.0
  y1 = 10.0

  msh = RectangleMesh(N, N, x1, y1)

  MC = MeshConstant(msh)
  dt = MC.Constant(10. / N)
  t = MC.Constant(0.0)

  V = FunctionSpace(msh, "CG", 1)
  x, y = SpatialCoordinate(msh)

  S = Constant(2.0)
  C = Constant(1000.0)
  B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
  R = (x * x + y * y) ** 0.5

  uexact = B * atan(t)*(pi / 2.0 - atan(S * (R - t)))
  rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))
  u = interpolate(uexact, V)

  v = TestFunction(V)
  F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v) * dx

  bc = DirichletBC(V, 0, "on_boundary")

Now, we define the solver parameters.  We get at the Rana method
through an Irksome-provided Python preconditioner.  This method
inherits from :class:`firedrake.AuxiliaryOperatorPC` (it provides the
Jacobian of the variational form with the approximate Butcher tableu
substituted) and so provides the user with an 'aux' preconditioner
to configure.  Since the Rana technique gives us a block triangular
matrix, a multiplicative field split exactly applies the preconditioner
if its diagonal blocks are exactly inverted::

  params = {"mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "gmres",
            "ksp_monitor": None,
            "pc_type": "python",
            "pc_python_type": "irksome.RanaLD",
	    "aux": {
	        "pc_type": "fieldsplit",
		"pc_fieldsplit_type": "multiplicative"
	    }}

But they don't have to be.  We'll approximate the inverse of each
diagonal block with a sweep of gamg (PETSc's multigrid)::

  per_field = {"ksp_type": "preonly",
               "pc_type": "gamg"}

  for s in range(butcher_tableau.num_stages):
      params["fieldsplit_%s" % (s,)] = per_field

Note that we have used the same technique for each RK stage, which is
probably typical.  However, it is not necessary at all.

To test this preconditioning strategy, we'll create a time stepping
object which will set up the variational problem for us.  (Important
note:  The stepper puts some special information into a PETSc context
for the variational problem it configures.  This is vital for the
Rana-type preconditioners to function.  If you want to use the
preconditioner outside of the :class:`.TimeStepper` then you will have
some extra setup to do)::

  stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                        solver_parameters=params)

But, since we're just testing the efficacy of the preconditioner,
we'll solve the inside variational problem one time::

  stepper.solver.solve()
