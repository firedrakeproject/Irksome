Solving monodomain equations with Fitzhugh-Nagumo reaction and an IMEX method
=============================================================================

We're solving monodomain (reaction-diffusion) with a particular reaction term::

  import copy

  from firedrake import (And, Constant, File, Function, FunctionSpace,
                         RectangleMesh, SpatialCoordinate, TestFunctions,
                         as_matrix, conditional, dx, grad, inner, split)
  from irksome import Dt, RadauIIA, TimeStepper

  mesh = RectangleMesh(100, 100, 70, 70, quadrilateral=True)
  polyOrder = 2

  # Set up the function space and test/trial functions.
  V = FunctionSpace(mesh, "S", polyOrder)
  Z = V * V

  x, y = SpatialCoordinate(mesh)
  dt = Constant(0.05)
  t = Constant(0.0)

  InitialPotential = conditional(x < 3.5, Constant(2.0), Constant(-1.28791))
  InitialCell = conditional(And(And(31 <= x, x < 39), And(0 <= y, y < 35)),
                            Constant(2.0), Constant(-0.5758))

  eps = Constant(0.1)
  beta = Constant(1.0)
  gamma = Constant(0.5)

  chi = Constant(1.0)
  capacitance = Constant(1.0)

  sigma1 = sigma2 = 1.0

  sigma = as_matrix([[sigma1, 0.0], [0.0, sigma2]])


  # Set up Irksome to be used.
  butcher_tableau = RadauIIA(2)
  ns = butcher_tableau.num_stages

  uu = Function(Z)
  vu, vc = TestFunctions(Z)
  uu.sub(0).interpolate(InitialPotential)
  uu.sub(1).interpolate(InitialCell)

  (uCurr, cCurr) = split(uu)

  F1 = (inner(chi * capacitance * Dt(uCurr), vu)*dx
        + inner(grad(uCurr), sigma * grad(vu))*dx
        + inner(Dt(cCurr), vc)*dx - inner(eps * uCurr, vc)*dx
        - inner(beta * eps, vc)*dx + inner(gamma * eps * cCurr, vc)*dx)

  F2 = inner((chi/eps) * (-uCurr + (uCurr**3 / 3) + cCurr), vu)*dx

  params = {"snes_type": "ksponly",
            "ksp_rtol": 1.e-8,
            "ksp_monitor": None,
            "mat_type": "matfree",
            "ksp_type": "fgmres",
            "pc_type": "python",
            "pc_python_type": "irksome.RanaLD",
            "aux": {
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "multiplicative"}}

  per_stage = {
      "ksp_type": "preonly",
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

  for s in range(ns):
      params["aux"][f"pc_fieldsplit_{s}_fields"] = f"{2*s},{2*s+1}"
      params["aux"][f"fieldsplit_{s}"] = per_stage

  itparams = copy.deepcopy(params)
  itparams["ksp_rtol"] = 1.e-4

  stepper = TimeStepper(F1, butcher_tableau, t, dt, uu,
                        stage_type="imex",
                        prop_solver_parameters=params,
                        it_solver_parameters=itparams,
		        Fexp=F2,
		        num_its_initial=5,
		        num_its_per_step=3)

  uFinal, cFinal = uu.split()
  outfile1 = File("FHN_results/FHN_2d_u.pvd")
  outfile2 = File("FHN_results/FHN_2d_c.pvd")
  outfile1.write(uFinal, time=0)
  outfile2.write(cFinal, time=0)

  for j in range(12):
      print(f"{float(t)}")
      stepper.advance()
      t.assign(float(t) + float(dt))
      # uCurr, cCurr = split(uu)
      if (j % 5 == 0):
          print("Time step", j)
          outfile1.write(uFinal, time=j * float(dt))
          outfile2.write(cFinal, time=j * float(dt))

  nsteps, nprop, nit, nnonlinprop, nlinprop, nnonlinit, nlinit = stepper.solver_stats()
  print(f"Time steps taken: {nsteps}")
  print(f"  {nprop} propagator steps")
  print(f"  {nit} iterator steps")
  print(f"  {nnonlinprop} nonlinear steps in propagators")
  print(f"  {nlinprop} linear steps in propagators")
  print(f"  {nnonlinit} nonlinear steps in iterators")
  print(f"  {nlinit} linear steps in iterators")  

