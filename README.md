# Irksome

This package works with Firedrake to generate Runge-Kutta methods from a semi-discrete UFL form.  We have added a UFL symbol for time derivatives and can produce UFL for the fully discrete method from a semi-discrete form and a Butcher tableau.  Several such tableaux are available, and some utility functions for time-stepping and adaptive time-stepping provided the tableau has an embedded lower-order method.

A long-standing critique of fully implicit RK methods, especially for PDE, is that they require a very large algebraic solve for all stages concurrently.  However, we can use Firedrake's solver infrastructure to address this issue, and also recover most of the comparative efficiency of DIRK or explicit methods.

The core of Irksome is based on UFL manipulation and so should be adaptable to work with FEniCS or other UFL-based packages, but the current version works only with Firedrake.  As such, it requires a working Irksome installation.  We recommend installing Irksome via the `--install irksome` option.  Given a preexisting Firedrake installation, one may obtain Irksome with options to `firedrake-update`.
