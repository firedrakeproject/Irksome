from firedrake import *
from fetsome.fet.fetutils import spacetime_dot
from fetsome.fet.lagrangebases import lagrange_bases
from numpy import dot, squeeze
from math import sqrt

def time_errornorm(u, uh, quadrature, dt, t, T):
    return general_time_errornorm(u, uh, quadrature, dt, t, T)

# Hard-coded implementation of error calculation for linear time elements
# for comparison and testing of general error
def linear_time_errornorm(u, uh, quadrature, dt, t):
    # Takes the error norm of a function with time quadrature when time FE is linear
    V = uh[0].function_space()
    e_squared = 0.
    for i in range(len(uh) - 1):
        u0 = uh[i]
        u1 = uh[i+1]
        psi = lagrange_bases["linear"]
        e_squared += dot([norm(interpolate(replace(u, {t: Constant((i + tau)*dt)}), V) - (psi[0](tau)*u0 + psi[1](tau)*u1))**2
                        for tau in quadrature.points], quadrature.weights) * dt

    return sqrt(e_squared)

# Hard-coded implementation of error calculation for quadratic time elements
# for comparison and testing of general error
def quadratic_time_errornorm(u, uh, quadrature, dt, t):
    # Takes the error norm of a function with time quadrature when time FE is quadratic
    V = uh[0].function_space()
    e_squared = 0.
    for i in range(0, len(uh) - 2, 2):
        u0 = uh[i]
        u1 = uh[i+1]
        u2 = uh[i+2]
        psi = lagrange_bases["quadratic"]
        e_squared += dot([norm(interpolate(replace(u, {t: Constant((i/2. + tau)*dt)}), V) - (psi[0](tau)*u0 + psi[1](tau)*u1 + psi[2](tau)*u2))**2
                         for tau in quadrature.points], quadrature.weights) * dt
        
    return sqrt(e_squared)

# Hard-coded implementation of error calculation for cubic time elements
# for comparison and testing of general error
def cubic_time_errornorm(u, uh, quadrature, dt, t):
    # Takes the error norm of a function with time quadrature when time FE is cubic
    V = uh[0].function_space()
    e_squared = 0.
    for i in range(0, len(uh) - 3, 3):
        u0 = uh[i]
        u1 = uh[i+1]
        u2 = uh[i+2]
        u3 = uh[i+3]
        psi = lagrange_bases["cubic"]
        e_squared += dot([norm(interpolate(replace(u, {t: Constant((i/3. + tau)*dt)}), V) - (psi[0](tau)*u0 + psi[1](tau)*u1 + psi[2](tau)*u2 + psi[3](tau)*u3))**2
                         for tau in quadrature.points], quadrature.weights) * dt

    return sqrt(e_squared)

# General spacetime error norm calculator, takes the time function space
def general_time_errornorm(u, uh, quadrature, dt, t, T):
    # The following is very inefficient code!!!! To change if there is a better way

    V = uh[0].function_space()
    Tfs = T.finat_element.fiat_equivalent
    kt = Tfs.degree()

    # Tabulate the time basis functions at the quadrature points
    psi = squeeze(Tfs.tabulate(0, quadrature.points)[(0,)].T)

    # Initialise the squared error and loop
    e_squared = 0.
    for i in range(0, len(uh) - kt, kt):
        # Slice the spatial solutions for the current time element and
        # calculate low interval bound
        us_element = uh[i:i + kt + 1]
        tn = i / kt * dt

        # Perform the quadrature sum over all points and weights
        for q in range(quadrature.num_points):
            # Tabulate the temporal basis functions
            bases_tau = psi[q]

            # Compose expression for the difference of functions
            u_tau_exact = interpolate(replace(u, {t: Constant(tn + quadrature.points[q] * dt)}), V)
            uh_tau = spacetime_dot(bases_tau, us_element)

            e_squared += norm(u_tau_exact - uh_tau)**2 * quadrature.weights[q] * dt

    return sqrt(e_squared)