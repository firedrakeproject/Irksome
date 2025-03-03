Mixed problems with the low-level interface
===========================================

Updating the solution at each time step is more complicated when we have
problems on mixed function spaces.  This demo peels back the :class:`TimeStepper` abstraction in the mixed heat equation demo.  In this case, the Dirichlet boundary conditions are weakly enforced.  However, mixed problems do not change the way strongly-enforced BC are handled, just how the solution is updated.

Imports::

  from irksome.tools import get_stage_space
  from firedrake import *
  from irksome import LobattoIIIC, Dt, getForm, MeshConstant
  from ufl.algorithms.ad import expand_derivatives

  butcher_tableau = LobattoIIIC(2)

Build the mesh and approximating spaces::

  N = 32
  x0 = 0.0
  x1 = 10.0
  y0 = 0.0
  y1 = 10.0
  msh = RectangleMesh(N, N, x1, y1)

  V = FunctionSpace(msh, "RT", 2)
  W = FunctionSpace(msh, "DG", 1)
  Z = V * W

  MC = MeshConstant(msh)
  dt = MC.Constant(10.0 / N)
  t = MC.Constant(0.0)

  x, y = SpatialCoordinate(msh)

  S = Constant(2.0)
  C = Constant(1000.0)

  B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
  R = (x * x + y * y) ** 0.5

  uexact = B * atan(t)*(pi / 2.0 - atan(S * (R - t)))
  sigexact = -grad(uexact)

  rhs = expand_derivatives(diff(uexact, t) + div(sigexact))

  sigu = project(as_vector([0, 0, uexact]), Z)
  sigma, u = split(sigu)

  v, w = TestFunctions(Z)

  F = (inner(Dt(u), w) * dx + inner(div(sigma), w) * dx - inner(rhs, w) * dx
       + inner(sigma, v) * dx - inner(u, div(v)) * dx)

Get the function space for the stage-coupled problem::

  Vbig = get_stage_space(Z, butcher_tableau.num_stages)
  k = Function(Vbig)

Get the form and new boundary conditions (which are dropped since
we have weak Dirichlet))::
  
  Fnew, _ = getForm(F, butcher_tableau, t, dt, sigu, k)

We set up the variational problem and solver using a sparse direct method::

  params = {"mat_type": "aij",
            "snes_type": "ksponly",
	    "ksp_type": "preonly",
            "pc_type": "lu"}

  prob = NonlinearVariationalProblem(Fnew, k)
  solver = NonlinearVariationalSolver(prob, solver_parameters=params)

Advancing the solution in time is a bit more complicated now::

  num_fields = len(Z)
  b = butcher_tableau.b

  while (float(t) < 1.0):
      if (float(t) + float(dt) > 1.0):
          dt.assign(1.0 - float(t))

      solver.solve()

      for s in range(butcher_tableau.num_stages):
          for i in range(num_fields):
	      sigu.dat.data[i][:] += float(dt) * b[s] * k.dat.data[num_fields * s + i][:]
  
      print(float(t))
      t.assign(float(t) + float(dt))


Finally, we check the accuracy of the solution::

  sigma, u = sigu.subfunctions
  print("U error      : ", errornorm(uexact, u) / norm(uexact))
  print("Sig error    : ", errornorm(sigexact, sigma) / norm(sigexact))
  print("Div Sig error: ",
        errornorm(sigexact, sigma, norm_type='Hdiv')
        / norm(sigexact, norm_type='Hdiv'))
