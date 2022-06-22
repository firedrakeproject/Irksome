from irksome.fetsome.estimatetimedegree import estimate_time_degree
import numpy as np
from math import sqrt
from FIAT import ufc_simplex
from FIAT.quadrature import GaussLegendreQuadratureLineRule, RadauQuadratureLineRule

class TimeQuadrature:
    # Time quadrature on the unit interval
    def __init__(self, points, weights):
        self.points = np.array(points).flatten()
        self.weights = np.array(weights).flatten()
        self.num_points = len(points)
    
    # Helper function to unwrap time function spaces to their FIAT form
    def _extract_fiat(self, T, Tprime):
        if not Tprime:
            # We assume that the test function spaces is the same as the trial function space
            Tprime = T
        
        Tfs = T.finat_element.fiat_equivalent
        Tprimefs = Tprime.finat_element.fiat_equivalent
        
        return Tfs, Tprimefs
    
    # Tabulates the basis functions of the time function space at the quadrature points
    def tabulate_at_points(self, T, order):
        Tfs = T.finat_element.fiat_equivalent
        tabulate = Tfs.tabulate(order, self.points)
        clean_tabulate = {}
        for key in tabulate:
            n, = key
            clean_tabulate[n] = tabulate[key]
        return clean_tabulate
    
    # Quadrature mass matrix
    def time_mass(self, T, Tprime=None):
        index = (0, 0)
        return self.time_quadrature_matrix_multi(index, T, Tprime)

    # Quadrature half-stiffness matrix, differentiating trial basis
    def time_half_stiffness_on_trial(self, T, Tprime=None):
        index = (1, 0)
        return self.time_quadrature_matrix_multi(index, T, Tprime)
    
    # Quadrature half-stiffness matrix, differentiating test basis
    def time_half_stiffness_on_test(self, T, Tprime=None):
        index = (0, 1)
        return self.time_quadrature_matrix_multi(index, T, Tprime)

    # Quadrature stiffness matrix
    def time_stiffness(self, T, Tprime=None):
        index = (1, 1)
        return self.time_quadrature_matrix_multi(index, T, Tprime)
    
    # Arbitrary
    def time_quadrature_matrix_multi(self, multi_index, T, Tprime=None):
        Tfs, Tprimefs = self._extract_fiat(T, Tprime)
        trial_order, test_order = multi_index

        psi_of_order = Tfs.tabulate(trial_order, self.points)[(trial_order,)]
        psi_prime_of_order = Tprimefs.tabulate(test_order, self.points)[(test_order,)]

        S = psi_of_order @ np.diag(self.weights) @ psi_prime_of_order.T
        return S

def time_gauss_quadrature():
    # 2nd degree (2 point) Gaussian quadrature
    points = [(1./2. - 1. / (2*sqrt(3))), (1./2. + 1. / (2*sqrt(3)))]
    weights = [1./2, 1./2.]
    return TimeQuadrature(points, weights)

def time_gauss_quadrature_little():
    #3rd degree (3 point) Gaussian quadrature
    points = [(-0.774597/2. + 1./2.), (0. + 1./2.), (0.774597/2. + 1./2.)]
    weights = [(0.555556/2.), (0.888889/2.), (0.555556/2.)]
    return TimeQuadrature(points, weights)

def time_gauss_quadrature_overkill():
    #4th degree (4 point) Gaussian quadrature
    points = [(-0.861135/2. + 1./2.), (-0.339981/2. + 1./2.), (0.339981/2. + 1./2.), (0.861135/2. + 1./2.)]
    weights = [(0.347855/2.), (0.652145/2.), (0.652145/2.), (0.347855/2.)]
    return TimeQuadrature(points, weights)

def time_gauss_quadrature_excessive():
    #5th degree (5 point) Gaussian quadrature
    points = [(-0.90618/2. + 1./2.), (-0.538469/2. + 1./2.), (0. + 1./2.),
              (0.538469/2. + 1./2.), (0.90618/2. + 1./2.)]
    weights = [(0.236927/2.), (0.478629/2.), (0.568889/2.), (0.478629/2.), (0.236927/2.)]
    return TimeQuadrature(points, weights)


# Utility function to estimate the order of Gaussian time quadrature needed for
# integration of time forms
def estimate_gauss_time_quadrature(F, t, kt):
    # Gaussian quadrature degree of precision is q = 2n - 1
    estimated = estimate_time_degree(F, t, kt)
    q = (estimated + 1 + 1)//2
    return q

# Utility function to estimate the order of Gauss-Radau time quadrature
def estimate_radau_time_quadrature(F, t, kt):
    # Radau quadrature degree of precision is q = 2n - 2
    estimated = estimate_time_degree(F, t, kt)
    q = (estimated + 2 + 1)//2
    return q

def make_gauss_time_quadrature(num_points):
    # Use FIAT to create a Gauss Legendre quadrature rule in 1 dimension (for time)
    interval = ufc_simplex(1)
    fiat_quadrature = GaussLegendreQuadratureLineRule(interval, num_points)
    return TimeQuadrature(fiat_quadrature.get_points(), fiat_quadrature.get_weights())

def make_radau_time_quadrature(num_points):
    # Use FIAT to create a Radau quadrature rule in 1 dimension (for time)
    interval = ufc_simplex(1)
    fiat_quadrature = RadauQuadratureLineRule(interval, num_points)
    return TimeQuadrature(fiat_quadrature.get_points(), fiat_quadrature.get_weights())