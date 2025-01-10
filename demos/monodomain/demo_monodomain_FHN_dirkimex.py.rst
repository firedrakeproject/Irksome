Solving monodomain equations with Fitzhugh-Nagumo reaction and a DIRK- IMEX method
==================================================================================

We're solving monodomain (reaction-diffusion) with a particular reaction term.
The basic form of the equation is:

.. math::

   \chi \left( C_m u_t + I_{ion}(u) \right) = \nabla \cdot \sigma \nabla u

where :math:`u` is the membrane potential, :math:`\sigma` is the conductivity tensor, :math:`C_m` is the specific capacitance of the cell membrane, and :math:`\chi` is the surface area to volume ration.  The term :math:`I_{ion}` is current due to ionic flows through channels in the cell membranes, and may couple to a complicated reaction network.  In our case, we take the relatively simple model due to Fitzhugh and Nagumo.  Here, we have a separate concentration variable :math:`c` satisfying the reaction equation:

.. math::

   c_t = \epsilon( u + \beta - \gamma c)

for certain positive parameters :math:`\beta` and :math:`\gamma`, and the current takes the form of:

.. math::

   I_{ion}(u, c) = \tfrac{1}{\epsilon} \left( u - \tfrac{u^3}{3} - c \right)

so that we have an overall system of two equations.  One of them is linear but stiff/diffusive, and the other is nonstiff but nonlinear.  This combination makes the system a good candidate for IMEX-type methods.


We start with standard Firedrake/Irksome imports::

  import copy

  from firedrake import (And, Constant, File, Function, FunctionSpace,
                         RectangleMesh, SpatialCoordinate, TestFunctions,
                         as_matrix, conditional, dx, grad, inner, split)
  from irksome import Dt, MeshConstant, IMEX4, TimeStepper

And we set up the mesh and function space.::
  
  mesh = RectangleMesh(20, 20, 70, 70, quadrilateral=True)
  polyOrder = 2
  
  V = FunctionSpace(mesh, "CG", 2)
  Z = V * V

  x, y = SpatialCoordinate(mesh)
  MC = MeshConstant(mesh)
  dt = MC.Constant(0.05)
  t = MC.Constant(0.0)

Specify the physical constants and initial conditions::

  eps = Constant(0.1)
  beta = Constant(1.0)
  gamma = Constant(0.5)

  chi = Constant(1.0)
  capacitance = Constant(1.0)

  sigma1 = sigma2 = 1.0
  sigma = as_matrix([[sigma1, 0.0], [0.0, sigma2]])

  
  InitialPotential = conditional(x < 3.5, Constant(2.0), Constant(-1.28791))
  InitialCell = conditional(And(And(31 <= x, x < 39), And(0 <= y, y < 35)),
                            Constant(2.0), Constant(-0.5758))


  uu = Function(Z)
  vu, vc = TestFunctions(Z)
  uu.sub(0).interpolate(InitialPotential)
  uu.sub(1).interpolate(InitialCell)

  (u, c) = split(uu)
  

This sets up the Butcher tableau.  All of our IMEX-type methods are
based on a RadauIIA method for the implicit part.  We use a three-stage method.::
  
  butcher_tableau = IMEX4()
  ns = butcher_tableau.num_stages

To access an IMEX method, we need to separately specify the implicit and explicit parts of the operator.
The part to be handled implicitly is taken to contain the time derivatives as well::
  
  F1 = (inner(chi * capacitance * Dt(u), vu)*dx
        + inner(grad(u), sigma * grad(vu))*dx
        + inner(Dt(c), vc)*dx - inner(eps * u, vc)*dx
        - inner(beta * eps, vc)*dx + inner(gamma * eps * c, vc)*dx)

This is the part to be handled explicitly.::
	  
  F2 = - inner((chi/eps) * (-u + (u**3 / 3) + c), vu)*dx

If we wanted to use a fully implicit method, we would just take
F = F1 - F2.   Note the minus sign, since DIRK-IMEX takes forms as F1 = F2.

Now, set up solver parameters.  Since we're using a DIRK-IMEX scheme, we can
specify only parameters for each stage::
  
  params = {"snes_type": "ksponly",
            "ksp_monitor": None,
            "mat_type": "aij",
            "ksp_type": "fgmres",
	    "pc_type": "fieldsplit",
	    "pc_fieldsplit_type": "additive",
	    "fieldsplit_0": {
                "ksp_type": "preonly",
                "pc_type": "gamg",
	    },
	    "fieldsplit_1": {
                "ksp_type": "preonly",
                "pc_type": "icc",
	    }}


The DIRK-IMEX schemes also require a mass-matrix solver.  Here, we just use AMG on the coupled system, which works fine.::

  mass_params = {"snes_type": "ksponly",
                 "ksp_rtol": 1.e-8,
		 "ksp_monitor": None,
		 "mat_type": "aij",
		 "ksp_type": "fgmres",
		 "pc_type": "gamg",
		}

Now, we access the IMEX method via the `TimeStepper` as with other methods.  Note that we specify somewhat different kwargs, needing to specify the implicit and explicit parts separately as well as separate solver options for the implicit and mass solvers.::
  
  stepper = TimeStepper(F1, butcher_tableau, t, dt, uu,
                        stage_type="dirkimex",
                        solver_parameters=params,
                        mass_parameters=mass_params,
		        Fexp=F2)

  uFinal, cFinal = uu.split()
  outfile1 = File("FHN_results/FHN_2d_u.pvd")
  outfile2 = File("FHN_results/FHN_2d_c.pvd")
  outfile1.write(uFinal, time=0)
  outfile2.write(cFinal, time=0)

  for j in range(12):
      print(f"{float(t)}")
      stepper.advance()
      t.assign(float(t) + float(dt))

      if (j % 5 == 0):
          outfile1.write(uFinal, time=j * float(dt))
          outfile2.write(cFinal, time=j * float(dt))

  nsteps, n_nonlin, n_lin, n_nonlin_mass, n_lin_mass = stepper.solver_stats()
  print(f"Time steps taken: {nsteps}")
  print(f"  {n_nonlin} nonlinear steps in implicit stage solves (should be {nsteps*ns})")
  print(f"  {n_lin} linear steps in implicit stage solves")
  print(f"  {n_nonlin_mass} nonlinear steps in mass solves (should be {nsteps*(ns+1)})")
  print(f"  {n_lin_mass} linear steps in mass solves")

