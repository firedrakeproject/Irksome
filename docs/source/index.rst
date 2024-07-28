Welcome to Irksome
==================

Irksome is a Python library that adds a temporal discretization layer on top of
finite element discretizations provided by `Firedrake <https://firedrakeproject.org>`__.  We provide a symbolic
representation of time derivatives in UFL, allowing users to write weak forms
of semidiscrete PDE.  Irksome maps this and a Butcher tableau encoding a Runge-Kutta method into a fully discrete variational problem for the stage values.  Irksome then leverages existing advanced solver technology in Firedrake and `PETSc <https://www.mcs.anl.gov/petsc/>`__ to allow for efficient computation of the Runge-Kutta stages.
Convenience classes package the underlying lower-level manipulations and present users with a friendly high-level interface time stepping.


So, instead of manually coding UFL for backward Euler for the heat equation::

  F = inner((unew - uold) / dt, v) * dx + inner(grad(unew), grad(v)) * dx

and rewriting this if you want a different time-stepping method, Irksome lets you write UFL for a semidiscrete form::
  
  F = inner(Dt(u), v) * dx + inner(grad(u), grad(v)) * dx

and maps this and a Butcher tableau for some Runge-Kutta method to UFL for a fully-discrete method.  Hence, switching between RK methods is plug-and-play. 

Irksome provides convenience classes to package the transformation of
forms and boundary conditions and provide a method to advance by a
time step.  The underlying variational problem for the (possibly
implicit!) Runge-Kutta stages composes fully with advanced
Firedrake/PETSc solver technology, so you can use block
preconditioners, multigrid with patch smooters, and more -- and in
parallel, too!
 


Getting started
===============

Irksome requires `Firedrake <https://www.firedrakeproject.org/>`__, if
you have an existing Firedrake installation, you can add Irksome to it
with ``firedrake-update --install irksome``. If you would like a fresh
installation including Irksome, use ``firedrake-install --install
irksome``. Alternately you can clone the repository and ``pip
install -e .`` into the Firedrake virtual environment.

Tutorials
=========
After your installation works, please check out our demos.


The best place to start are with some simple heat and wave equations:

.. toctree::
   :maxdepth: 1

   demos/demo_heat.py
   demos/demo_mixed_heat.py
   demos/demo_RTwave.py

Since those demos invariably rely on the non-scalable LU factorization,
we have several demos showing how to work with Firedrake solver options
to deploy more efficient methods:

.. toctree::
   :maxdepth: 1

   demos/demo_heat_pc.py
   demos/demo_heat_mg.py
   demos/demo_heat_rana.py
   demos/demo_dirk_parameters.py

We now have support for DIRKs:

.. toctree::
   :maxdepth: 1

   demos/demo_heat_dirk.py

Or check out an IMEX-type method for the monodomain equations:

.. toctree::
   :maxdepth: 1

   demos/demo_monodomain_FHN.py

Advanced demos
--------------

There's also an example solving a Sobolev-type equation

.. toctree::
   :maxdepth: 1

   demos/demo_bbm.py

Finally, if you feel you must bypass the :py:class:`.TimeStepper`
abstraction, we have some examples how to interact with Irksome at a
slightly lower level:

.. toctree::
   :maxdepth: 1

   demos/demo_lowlevel_homogbc.py
   demos/demo_lowlevel_inhomogbc.py   
   demos/demo_lowlevel_mixed_heat.py

API documentation
=================

There is also an alphabetical :ref:`index <genindex>`, and a
:ref:`search engine <search>`.

.. toctree::
   :maxdepth: 2

   irksome
