# Irksome

This package works with Firedrake to generate Runge-Kutta methods from a semi-discrete UFL form.  We have added a UFL symbol for time derivatives and can produce UFL for the fully discrete method from a semi-discrete form and a Butcher tableau.  Several such tableaux are available, and some utility functions for time-stepping and adaptive time-stepping provided the tableau has an embedded lower-order method.

A long-standing critique of fully implicit RK methods, especially for PDE, is that they require a very large algebraic solve for all stages concurrently.  However, we can use Firedrake's solver infrastructure to address this issue, and also recover most of the comparative efficiency of DIRK or explicit methods.

The core of Irksome is based on UFL manipulation and so should be adaptable to work with FEniCS or other UFL-based packages, but the current version works only with Firedrake.  As such, it requires a working Irksome installation.

To install Irksome you need a working Firedrake installation (instructions can be found [here](https://www.firedrakeproject.org/install.html)) and then Irksome can be installed by running:
```
$ pip install --src . --editable git+https://github.com/firedrakeproject/Irksome.git#egg=Irksome
```
or, equivalently:
```
$ git clone https://github.com/firedrakeproject/Irksome.git
$ pip install --editable ./Irksome
```
