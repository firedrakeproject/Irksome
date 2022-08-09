Solving unsteady Navier-Stokes with an IMEX method
==================================================

We are doing the classical lid-driven cavity on
:math:`\Omega=[0,1]^2, solving for velocity :math:`u` and pressure :math:`p` such that

.. math::

   u_t + u \cdot \nabla u - \frac{1}{Re} \Delta u + \nabla p & = 0

   \nabla \cdot u & = 0

together with Dirichlet boundary conditions putting :math:`u=(0,0)` on the left and right sides and bottom of the box and :math:`u=(1,0)` on the top of the box.

Some boilerplate imports plus setting up the domain and function spaces.  It's just Taylor-Hood, so feel free to do better if you wish::

  from irksome import Dt, RadauIIA, RadauIIAIMEXMethod

  from firedrake import (Constant, DirichletBC, File, Function, FunctionSpace,
                         MixedVectorSpaceBasis, TestFunctions, UnitSquareMesh,
                         VectorFunctionSpace, VectorSpaceBasis, assemble, div,
                         dot, dx, grad, inner, norm, split)

  N = 32
  M = UnitSquareMesh(N, N)

  V = VectorFunctionSpace(M, "CG", 2)
  W = FunctionSpace(M, "CG", 1)
  Z = V * W

  up = Function(Z)
  u, p = split(up)
  v, q = TestFunctions(Z)

  Re = Constant(100.0)

Now we set up two different variational forms.  The first one is the part of the
problem we wish to handle implicitly::

  Fimp = (inner(Dt(u), v) * dx
          + 1.0 / Re * inner(grad(u), grad(v)) * dx
          - p * div(v) * dx
          + div(u) * q * dx)

while the second is the part we "split off" to handle explicitly::

  Fexp = inner(dot(grad(u), u), v) * dx

Here are the standard boundary conditions and null space::

  bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
         DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

  nullspace = [(1, VectorSpaceBasis(constant=True))]

Here is the Butcher tableau.  It has to be a RadauIIA, but you can
experiment with the number of stages as desired::

  bt = RadauIIA(3)

Solver parameters are a thing.  This set of options applies the strategy
of Rana/Howle/et al to obtain a block triangular preconditioner.  In our case,
we are doing a relatively small problem so just hit the blocks with a direct
solver, but this can be upgraded to augmented Lagrangian or such as desired::

  if bt.num_stages > 1:
      solver_parameters = {
          "mat_type": "aij",
          "snes_type": "ksponly",
          "ksp_type": "gmres",
          "ksp_gmres_modifiedgramschmidt": None,
          "ksp_monitor": None,
          "ksp_converged_reason": None,
          "ksp_rtol": 1.e-10,
          "ksp_atol": 1.e-10,
          "pc_type": "python",
          "pc_python_type": "irksome.RanaLD",
          "aux": {
              "pc_type": "fieldsplit",
              "pc_fieldsplit_type": "multiplicative",
          }
      }
      per_stage = {
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"
      }
      spaux = solver_parameters["aux"]
      for s in range(bt.num_stages):
          spaux[f"pc_fieldsplit_{s}_fields"] = f"{2*s},{2*s+1}"
          spaux[f"fieldsplit_{s}"] = per_stage
  else:
      solver_parameters = {
          "mat_type": "aij",
          "snes_type": "ksponly",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"
      }

Now for setting up time and the stepper itself.  Notice that the
stepper constructor requires both the implicit and explicit forms,
whereas the other steppers just take a single form as input::

  t = Constant(0.0)
  dt = Constant(1/N)
  stepper = RadauIIAIMEXMethod(Fimp, Fexp, bt,
                               t, dt, up,
                               bcs=bcs,
                               prop_solver_parameters=solver_parameters,
                               it_solver_parameters=solver_parameters,
                               nullspace=nullspace)

The IMEX methods have a concept of "propagators" and "iterators".
One needs to use the later to get the values of the solution in the
first time slab, which are initially rubbish except for the initial condition.::

  num_iter_init = 10
  print("iterating")
  for i in range(num_iter_init):
      stepper.iterate()

Additionally, after taking a time step using the propagator, one may optionally
apply the iterator a few times.  This can improve the stability and/or accuracy
of the method.  In the limit as the number of iterations approaches infinity,
the IMEX method should converge to the solution of the fully implicit RadauIIA
method (if it converges...)::

  num_iter_perstep = 0
  u, p = up.split()
  print("time stepping")
  while (float(t) < 1.0):
      if (float(t) + float(dt) > 1.0):
          dt.assign(1.0 - float(t))
      stepper.advance()
      p -= assemble(p*dx)
      for i in range(num_iter_perstep):
          stepper.iterate()
          p -= assemble(p*dx)
      t.assign(float(t) + float(dt))
      print(norm(u), norm(p), assemble(p * dx))

Write the finally computed solution to disk::

  File("nseimex.pvd").write(u, p)
