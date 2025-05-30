from firedrake import *
from irksome import Dt, GalerkinTimeStepper, TimeStepper, GaussLegendre
from irksome.galerkin_stepper import TimeProjector
from irksome.labeling import TimeQuadratureLabel
from FIAT import ufc_simplex
from FIAT.quadrature_schemes import create_quadrature
import math
from scipy import special
from firedrake.petsc import PETSc

print = PETSc.Sys.Print

ufcline = ufc_simplex(1)

'''
Thanks to Boris Andrews for providing the Hill vortex functions
'''
'''
Hill vortex functions
'''
# UFL-compatible Bessel function
besselJ = bessel_J

def besselJ(alpha, x, layers=10):
    return sum([
        (-1)**m / math.factorial(m) / special.gamma(m + alpha + 1)
      * (x/2)**(2*m+alpha)
        for m in range(layers)
    ])


# Bessel function parameters
besselJ_root = 5.7634591968945506
besselJ_root_threehalves = besselJ(3/2, besselJ_root)


# (r, theta, phi) components of Hill vortex
def hill_r(r, theta, radius):
    rho = r / radius
    return 2 * (
        besselJ(3/2, besselJ_root*rho) / rho**(3/2)
      - besselJ_root_threehalves
    ) * cos(theta)


def hill_theta(r, theta, radius):
    rho = r / radius
    return (
        besselJ_root * besselJ(5/2, besselJ_root*rho) / rho**(1/2)
      + 2 * besselJ_root_threehalves
      - 2 * besselJ(3/2, besselJ_root*rho) / rho**(3/2)
    ) * sin(theta)


def hill_phi(r, theta, radius):
    rho = r / radius
    return besselJ_root * (
        besselJ(3/2, besselJ_root*rho) / rho**(3/2)
      - besselJ_root_threehalves
    ) * rho * sin(theta)


# Hill vortex (Cartesian)
def hill(vec, radius):
    (x, y, z) = vec
    rho = sqrt(x*x + y*y)

    r = sqrt(dot(vec, vec))
    theta = pi/2-atan2(z, rho)
    phi = atan2(y, x)

    r_dir = as_vector([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])
    theta_dir = as_vector([cos(phi)*cos(theta), sin(phi)*cos(theta), -sin(theta)])
    phi_dir = as_vector([-sin(phi), cos(phi), 0])

    return conditional(  # If we're outside the vortex...
        ge(r, radius),
        zero((3,)),
        conditional(  # If we're at the origin...
            le(r, 1e-13),
            as_vector([0, 0, 2*((besselJ_root/2)**(3/2)/special.gamma(5/2) - besselJ_root_threehalves)]),
            (hill_r(r, theta, radius) * r_dir
             + hill_theta(r, theta, radius) * theta_dir
             + hill_phi(r, theta, radius) * phi_dir)
        )
    )


def stokes_pair(msh):
    V = VectorFunctionSpace(msh, "CG", 2)
    Q = FunctionSpace(msh, "CG", 1)
    return V, Q


def nse_naive(msh, order, t, dt, Re, solver_parameters=None):
    hill_expr = hill(SpatialCoordinate(msh), 0.25)
    V, Q = stokes_pair(msh)
    Z = V * Q
    up = Function(Z)
    up.subfunctions[0].interpolate(hill_expr)
    # VTKFile("output/hill.pvd").write(up.subfunctions[0])

    v, q = TestFunctions(Z)
    u, p = split(up)

    Qhigh = create_quadrature(ufcline, 3*order-1)
    Qlow = create_quadrature(ufcline, 2*(order-1))
    Lhigh = TimeQuadratureLabel(Qhigh.get_points(), Qhigh.get_weights())
    Llow = TimeQuadratureLabel(Qlow.get_points(), Qlow.get_weights())

    F = (Llow(inner(Dt(u), v) * dx)
         + Lhigh(-inner(cross(u, curl(u)), v) * dx)
         + 1/Re * inner(grad(u), grad(v)) * dx
         - inner(p, div(v)) * dx
         - inner(div(u), q) * dx)

    bcs = DirichletBC(Z.sub(0), 0, "on_boundary")

    stepper = GalerkinTimeStepper(F, order, t, dt, up, bcs=bcs)
    Q1 = inner(u, u) * dx
    Q2 = inner(u, curl(u)) * dx
    invariants = [Q1, Q2]
    return stepper, invariants


def nse_aux_variable(msh, order, t, dt, Re, solver_parameters=None):
    hill_expr = hill(SpatialCoordinate(msh), 0.25)
    V, Q = stokes_pair(msh)
    Z = V * V * V * Q
    up = Function(Z)
    up.subfunctions[0].interpolate(hill_expr)

    v, v1, v2, q = TestFunctions(Z)
    u, w1, w2, p = split(up)

    Qhigh = create_quadrature(ufcline, 3*(order-1))
    Qlow = create_quadrature(ufcline, 2*(order-1))
    Lhigh = TimeQuadratureLabel(Qhigh.get_points(), Qhigh.get_weights())
    Llow = TimeQuadratureLabel(Qlow.get_points(), Qlow.get_weights())

    F = (Llow(inner(Dt(u), v) * dx)
         + Lhigh(-inner(cross(w1, w2), v) * dx)
         + Llow(1/Re * inner(grad(w1), grad(v)) * dx)
         - inner(p, div(v)) * dx
         - inner(div(u), q) * dx
         + inner(w1 - u, v1) * dx
         + inner(w2 - curl(u), v2) * dx
         )

    bcs = [DirichletBC(Z.sub(i), 0, "on_boundary") for i in range(len(Z)-1)]

    stepper = GalerkinTimeStepper(F, order, t, dt, up, bcs=bcs, aux_indices=(1, 2))
    Q1 = inner(u, u) * dx
    Q2 = inner(u, curl(u)) * dx
    invariants = [Q1, Q2]
    return stepper, invariants


def nse_project(msh, order, t, dt, Re, solver_parameters=None):
    hill_expr = hill(SpatialCoordinate(msh), 0.25)
    V, Q = stokes_pair(msh)
    Z = V * Q
    up = Function(Z)
    up.subfunctions[0].interpolate(hill_expr)

    v, q = TestFunctions(Z)
    u, p = split(up)

    Qhigh = create_quadrature(ufcline, 3*(order-1))
    Qlow = create_quadrature(ufcline, 2*(order-1))
    Lhigh = TimeQuadratureLabel(Qhigh.get_points(), Qhigh.get_weights())
    Llow = TimeQuadratureLabel(Qlow.get_points(), Qlow.get_weights())

    Qproj = create_quadrature(ufcline, 2*order)
    w1 = TimeProjector(u, order-1, Qproj)
    w2 = TimeProjector(curl(u), order-1, Qproj)

    F = (Llow(inner(Dt(u), v) * dx)
         + Lhigh(-inner(cross(w1, w2), v) * dx)
         + Llow(1/Re * inner(grad(w1), grad(v)) * dx)
         - inner(p, div(v)) * dx
         - inner(div(u), q) * dx
         )

    bcs = DirichletBC(Z.sub(0), 0, "on_boundary")

    stepper = GalerkinTimeStepper(F, order, t, dt, up, bcs=bcs)
    Q1 = inner(u, u) * dx
    Q2 = inner(u, curl(u)) * dx
    invariants = [Q1, Q2]
    return stepper, invariants


def run_nse(stepper, invariants):
    t = stepper.t
    dt = stepper.dt
    row = [float(t), *map(assemble, invariants)]
    print(*(f"{r:.8e}" for r in row))
    while float(t) < 3 * 2**(-6):
    # for k in range(2):
        stepper.advance()
        t.assign(t + dt)

        row = [float(t), *map(assemble, invariants)]
        print(*(f"{r:.8e}" for r in row))


order = 2
N = 8
msh = UnitCubeMesh(N, N, N)
msh.coordinates.dat.data[:, :] -= 0.5

t = Constant(0)
dt = Constant(2**-10)
Re = Constant(2**16)

solvers = {
    #"naive": nse_naive,
    "project": nse_project,
    #"aux": nse_aux_variable,
}

for name, solver in solvers.items():
    print(name)
    t.assign(0)
    run_nse(*solver(msh, order, t, dt, Re))
