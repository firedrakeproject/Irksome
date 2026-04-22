Solving the Mixed form of the Heat Equation
===========================================

This shows a different variational formulation of the heat equation.
As before, we put :math:`\Omega = [0,10] \times [0,10]`, with boundary :math:`\Gamma`: but now we write the heat equation in mixed form:

.. math::

   \sigma + \nabla u & = 0

   u_t + \nabla \cdot \sigma & = f


which gives rise to the weak form

.. math::

   (\sigma, v) - (u, \nabla \cdot v) &= 0

   (u_t, w) + (\nabla \cdot \sigma, w) &= (f, w)

Here, :math:`\sigma` is a vector-valued variable and will be discretized in
a suitable :math:`H(\mathrm{div})` -conforming space.  The variable :math:`u`
actually only requires :math:`L^2` regularity and will be discretized with
discontinuous piecewise polynomials.

Note that this gives us a differential-algebraic system at the fully discrete level as there is no time derivative on :math:`\sigma`.

Standard imports, although we're using a different RK scheme this time::

  from firedrake import *
  from irksome import LobattoIIIC, Dt, MeshConstant, TimeStepper

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

Create time and time-step variables::

  MC = MeshConstant(msh)
  dt = MC.Constant(10.0 / N)
  t = MC.Constant(0.0)

As in the first heat demo, we build the RHS via the method of
manufactured solutions::

  x, y = SpatialCoordinate(msh)

  S = Constant(2.0)
  C = Constant(1000.0)

  B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
  R = (x * x + y * y) ** 0.5

  uexact = B * atan(t)*(pi / 2.0 - atan(S * (R - t)))
  sigexact = -grad(uexact)

  rhs = Dt(uexact) + div(sigexact)


Set up the initial condition::

  sigu = project(as_vector([0, 0, uexact]), Z)
  sigma, u = split(sigu)

And define the variational form::

  v, w = TestFunctions(Z)

  F = (inner(Dt(u), w) * dx + inner(div(sigma), w) * dx - inner(rhs, w) * dx
       + inner(sigma, v) * dx - inner(u, div(v)) * dx)

As before, we use a sparse direct method::

  params = {"mat_type": "aij",
            "snes_type": "ksponly",
	    "ksp_type": "preonly",
            "pc_type": "lu"}

We set the time stepper as before, except there are no
strongly-enforced boundary conditiuons for the mixed method::

  stepper = TimeStepper(F, butcher_tableau, t, dt, sigu,
                        solver_parameters=params)

And we advance the solution in time::

  while (float(t) < 1.0):
      if (float(t) + float(dt) > 1.0):
          dt.assign(1.0 - float(t))
      stepper.advance()
      print(float(t))
      t.assign(float(t) + float(dt))

Finally, we check the accuracy of the solution::

  sigma, u = sigu.subfunctions
  print("U error      : ", errornorm(uexact, u) / norm(uexact))
  print("Sig error    : ", errornorm(sigexact, sigma) / norm(sigexact))
  print("Div Sig error: ",
        errornorm(sigexact, sigma, norm_type='Hdiv')
        / norm(sigexact, norm_type='Hdiv'))
