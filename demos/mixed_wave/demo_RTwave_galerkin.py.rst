Solving the Mixed Wave Equation: Energy conservation
====================================================

Let :math:`\Omega` be the unit square with boundary :math:`\Gamma`.  We write
the wave equation as a first-order system of PDE:

.. math::

   u_t + \nabla p & = 0

   p_t + \nabla \cdot u & = 0

together with homogeneous Dirichlet boundary conditions

.. math::

   p = 0 \quad \textrm{on}\ \Gamma

In this form, at each time, :math:`u` is a vector-valued function in the Soboleve space :math:`H(\mathrm{div})` and `p` is a scalar-valued function.  If we select appropriate test functions :math:`v` and :math:`w`, then we can arrive at the weak form

.. math::

   (u_t, v) - (p, \nabla \cdot v) & = 0

   (p_t, w) + (\nabla \cdot u, w) & = 0

Note that in mixed formulations, the Dirichlet boundary condition is weakly
enforced via integration by parts rather than strongly in the definition of
the approximating space.

In this example, we will use the next-to-lowest order Raviart-Thomas elements
for the velocity variable :math:`u` and discontinuous piecewise linear
polynomials for the scalar variable :math:`p`.

Here is some typical Firedrake boilerplate and the construction of a simple
mesh and the approximating spaces.  We are going to use a multigrid preconditioner for each timestep, so we create a MeshHierarchy as well::

  from firedrake import *
  from irksome import Dt, MeshConstant, GalerkinTimeStepper

  N = 10

  base = UnitSquareMesh(N, N)
  mh = MeshHierarchy(base, 2)
  msh = mh[-1]
  V = FunctionSpace(msh, "RT", 2)
  W = FunctionSpace(msh, "DG", 1)
  Z = V*W

Now we can build the initial condition, which has zero velocity and a sinusoidal displacement::

  x, y = SpatialCoordinate(msh)
  up0 = project(as_vector([0, 0, sin(pi*x)*sin(pi*y)]), Z)
  u0, p0 = split(up0)


We build the variational form in UFL::

  v, w = TestFunctions(Z)
  F = inner(Dt(u0), v)*dx + inner(div(u0), w) * dx + inner(Dt(p0), w)*dx - inner(p0, div(v)) * dx

Energy conservation is an important principle of the wave equation, and we can
test how well the spatial discretization conserves energy by creating a
UFL expression and evaluating it at each time step::

  E = 0.5 * (inner(u0, u0)*dx + inner(p0, p0)*dx)

The time and time step variables::

  MC = MeshConstant(msh)
  t = MC.Constant(0.0)
  dt = MC.Constant(1.0/N)

Here, we experiment with a multigrid preconditioner for the CG(2)-in-time discretization::

  mgparams = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "fgmres",
	      "ksp_rtol": 1e-8,
              "pc_type": "mg",
              "mg_levels": {
                  "ksp_type": "chebyshev",
                  "ksp_max_it": 2,
                  "ksp_convergence_test": "skip",
                  "pc_type": "python",
                  "pc_python_type": "firedrake.PatchPC",
                  "patch": {
                      "pc_patch": {
                          "save_operators": True,
                          "partition_of_unity": False,
                          "construct_type": "star",
                          "construct_dim": 0,
                          "sub_mat_type": "seqdense",
                          "dense_inverse": True,
                          "precompute_element_tensors": None},
                      "sub": {
                          "ksp_type": "preonly",
                          "pc_type": "lu"}}},
              "mg_coarse": {
                  "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps"}
              }
  

  stepper = GalerkinTimeStepper(F, 2, t, dt, up0,
                                solver_parameters=mgparams)


And, as with the heat equation, our time-stepping logic is quite simple.  At easch time step, we print out the energy in the system::

  print("Time    Energy")
  print("==============")

  while (float(t) < 1.0):
      if float(t) + float(dt) > 1.0:
          dt.assign(1.0 - float(t))

      stepper.advance()

      t.assign(float(t) + float(dt))
      print("{0:1.1e} {1:5e}".format(float(t), assemble(E)))

If all is well with the world, the energy will be nearly identical (up
to roundoff error) at each time step because the Galerkin-in-time methods
are symplectic and applied to a linear Hamiltonian system.

We can also confirm that the multigrid preconditioner is effective, by computing the average number of linear iterations per time-step::

  (steps, nl_its, linear_its) = stepper.solver_stats()
  print(f"The average number of multigrid iterations per time-step is {linear_its/steps}.")
