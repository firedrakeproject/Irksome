Welcome to Irksome
==================

Irksome is a Python library that adds a temporal discretization layer on top of
finite element discretizations provided by Firedrake.  We provide a symbolic
representation of time dericvatives in UFL, allowing users to write weak forms
of semidiscrete PDE.  Irksome maps this and a Butcher tableau encoding a Runge-Kutta method Runge-Kutta method into a fully discrete variational problem for the stage values.  Irksome then leverages existing advanced solver technology in Firedrake and PETSc to allow for efficient computation of the Runge-Kutta stages.
Convenience classes package the underlying lower-level manipulations and present users with a friendly high-level interface time stepping.

Getting started
===============

Irksome assumes you have a working Firedrake installation.  Within your Firedrake virtual environment, you can just clone the repository and ``pip install -e .``
within the top-level directory.  In the near future, we will have Irksome available as part of the Firedrake installation/updating mechanism so that
``firedrake-install --install irksome`` or
``firedrake-update --install irksome`` will download and install it for you.

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
   demos/demo_dirk_parameters.py

If you feel you must bypass the :class:`TimeStepper` abstraction, we have
some examples how to interact with Irksome at a slightly lower level:

.. toctree::
   :maxdepth: 1

   demos/demo_lowlevel_homogbc.py
   demos/demo_lowlevel_inhomogbc.py   
   demos/demo_lowlevel_mixed_heat.py

Module documentation is found :doc:`irksome` here.
