from firedrake import *
from irksome import Dt, GalerkinTimeStepper, TimeStepper, GaussLegendre
from irksome.galerkin_stepper import TimeProjector
from irksome.labeling import TimeQuadratureLabel
from FIAT import make_quadrature, ufc_simplex
import math
from scipy import special

'''
Thanks to Boris Andrews for providing the Hill vortex functions
'''
'''
Hill vortex functions
'''
# UFL-compatible Bessel function
def besselJ(x, alpha, layers=10):
    return sum([
        (-1)**m / math.factorial(m) / special.gamma(m + alpha + 1)
      * (x/2)**(2*m+alpha)
        for m in range(layers)
    ])



# Bessel function parameters
besselJ_root = 5.7634591968945506
besselJ_root_threehalves = besselJ(besselJ_root, 3/2)



# (r, theta, phi) components of Hill vortex
def hill_r(r, theta, radius):
    rho = r / radius
    return 2 * (
        besselJ(besselJ_root*rho, 3/2) / rho**(3/2)
      - besselJ_root_threehalves
    ) * cos(theta)

def hill_theta(r, theta, radius):
    rho = r / radius
    return (
        besselJ_root * besselJ(besselJ_root*rho, 5/2) / rho**(1/2)
      + 2 * besselJ_root_threehalves
      - 2 * besselJ(besselJ_root*rho, 3/2) / rho**(3/2)
    ) * sin(theta)

def hill_phi(r, theta, radius):
    rho = r / radius
    return besselJ_root * (
        besselJ(besselJ_root*rho, 3/2) / rho**(3/2)
      - besselJ_root_threehalves
    ) * rho * sin(theta)



# Hill vortex (Cartesian)
def hill(vec, radius):
    (x, y, z) = vec

    # Cylindrical/spherical coordinates
    r_cyl = sqrt(x**2 + y**2)  # Cylindrical radius
    r_sph = sqrt(dot(vec, vec))  # Spherical radius
    theta = pi/2 - atan2(z, r_cyl)
    rvec = vec / r_sph
    Rvec = as_vector([-y, x, 0]) / r_cyl
    THvec = as_vector([x*z, y*z, -r_cyl**2]) / r_sph / r_cyl

    return conditional(  # If we're outside the vortex...
        ge(r_sph, radius),
        zero((3,)),
        conditional(  # If we're at the origin...
            le(r_sph, 1e-13),
            as_vector([0, 0, 2*((besselJ_root/2)**(3/2)/special.gamma(5/2) - besselJ_root_threehalves)]),
            conditional(  # If we're on the z axis...
                le(r_cyl, 1e-13),
                as_vector([0, 0, hill_r(r_sph, 0, radius)]),
                (  # Else...
                    hill_r(r_sph, theta, radius) * rvec
                  + hill_theta(r_sph, theta, radius) * THvec
                  + hill_phi(r_sph, theta, radius) * Rvec
                )
            )
        )
    )


def nse_naive(msh, order, t, dt, Re, solver_parameters=None):
    hill_expr = hill(SpatialCoordinate(msh), 0.25)
    V = VectorFunctionSpace(msh, "CG", 2)
    W = FunctionSpace(msh, "CG", 1)
    Z = V * W
    up = Function(Z)
    up.subfunctions[0].interpolate(hill_expr)

    v, w = TestFunctions(Z)
    u, p = split(up)

    F = (inner(Dt(u), v) * dx
         - inner(cross(u, curl(u)), v) * dx
         + 1/Re * inner(grad(u), grad(v)) * dx
         - inner(p, div(v)) * dx
         + inner(div(u), w) * dx)

    bcs = DirichletBC(Z.sub(0), Constant((0, 0, 0)), "on_boundary")

    stepper = GalerkinTimeStepper(F, 2, t, dt, up, bcs=bcs)

    Q1 = inner(u, u) * dx
    Q2 = inner(u, curl(u)) * dx

    Q1s = [assemble(Q1)]
    Q2s = [assemble(Q2)]

    print(f"{float(t):.4e}, {assemble(Q1):.4e}, {assemble(Q2):.4e}")
    while float(t) < 3 * 2**(-6):
        stepper.advance()
        t.assign(float(t) + float(dt))
        Q1s.append(assemble(Q1))
        Q2s.append(assemble(Q2))
        print(f"{float(t):.4e}, {assemble(Q1):.4e}, {assemble(Q2):.4e}")

    return np.array(Q1s), np.array(Q2s)


def nse_project_both(msh, order, t, dt, Re, solver_parameters=None):
    hill_expr = hill(SpatialCoordinate(msh), 0.25)
    V = VectorFunctionSpace(msh, "CG", 2)
    W = FunctionSpace(msh, "CG", 1)
    Z = V * W
    up = Function(Z)
    up.subfunctions[0].interpolate(hill_expr)

    v, w = TestFunctions(Z)
    u, p = split(up)

    ufcline = ufc_simplex(1)

    Qhigh = make_quadrature(ufcline, 3*order-1)
    Qlow = make_quadrature(ufcline, 2*(order-1))

    Lhigh = TimeQuadratureLabel(Qhigh.get_points(), Qhigh.get_weights())
    Llow = TimeQuadratureLabel(Qlow.get_points(), Qlow.get_weights())

    Qk = make_quadrature(ufcline, order)
    w1 = TimeProjector(u, order-1, Qk)
    w2 = TimeProjector(curl(u), order-1, Qk)

    F = (Llow(inner(Dt(u), v) * dx) +
         Lhigh(- inner(cross(w1, w2), v) * dx) + (
         + 1/Re * inner(grad(w1), grad(v)) * dx
         - inner(p, div(v)) * dx
         + inner(div(u), w) * dx))

    bcs = DirichletBC(Z.sub(0), Constant((0, 0, 0)), "on_boundary")

    stepper = GalerkinTimeStepper(F, 2, t, dt, up, bcs=bcs)

    Q1 = inner(u, u) * dx
    Q2 = inner(u, curl(u)) * dx

    Q1s = [assemble(Q1)]
    Q2s = [assemble(Q2)]

    print(f"{float(t):.4e}, {assemble(Q1):.4e}, {assemble(Q2):.4e}")
    while float(t) < 3 * 2**(-6):
        stepper.advance()
        t.assign(float(t) + float(dt))
        Q1s.append(assemble(Q1))
        Q2s.append(assemble(Q2))
        print(f"{float(t):.4e}, {assemble(Q1):.4e}, {assemble(Q2):.4e}")

    return np.array(Q1s), np.array(Q2s)



N = 8
msh = UnitCubeMesh(N, N, N)
t = Constant(0)
dt = Constant(2**-10)
msh.coordinates.dat.data[:, :] -= 0.5
Re = Constant(2**8)
Q1s, Q2s = nse_naive(msh, 2, t, dt, Re)
print(Q1s)
print(Q2s)


