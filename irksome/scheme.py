from FIAT import ufc_simplex, create_quadrature
from FIAT.quadrature import RadauQuadratureLineRule


ufc_line = ufc_simplex(1)


class GalerkinScheme:
    """
    Base class for describing Galerkin-in-time methods in lieu of
    a Butcher tableau.

    :arg order: An integer indicating the order of the method
    :kwarg basis_type: A string indicating the finite element family (either
        `'Lagrange'` or `'Bernstein'`) or the Lagrange variant (either
        `'equispaced'`, `'spectral'`, `'chebyshev'`, or `'integral'`) for the
        test/trial spaces.  Defaults to equispaced Lagrange elements.
    :kwarg quadrature_degree: An integer indicating the degree of the
        quadrature to be use in time. Defaults to the sum
        of the degrees of the trial and test spaces.
    :kwarg quadrature_scheme: A string indicating the quadrature scheme
        to be used in time. Defaults to Gauss-Legendre.
    """
    def __init__(self, order,
                 basis_type=None,
                 quadrature_degree=None,
                 quadrature_scheme="default"):
        self.order = order
        self.basis_type = basis_type
        self.quadrature_degree = quadrature_degree
        self.quadrature_scheme = quadrature_scheme


class DiscontinuousGalerkinScheme(GalerkinScheme):
    """Class for describing DG-in-time methods"""
    def __init__(self, order,
                 basis_type=None,
                 quadrature_degree=None,
                 quadrature_scheme="default"):
        assert order >= 0, f"{type(self).__name__} must have order >= 0"
        super().__init__(order, basis_type=basis_type,
                         quadrature_degree=quadrature_degree,
                         quadrature_scheme=quadrature_scheme)


class ContinuousPetrovGalerkinScheme(GalerkinScheme):
    """Class for describing cPG-in-time methods"""
    def __init__(self, order,
                 basis_type=None,
                 quadrature_degree=None,
                 quadrature_scheme="default"):
        assert order >= 1, f"{type(self).__name__} must have order >= 1"
        super().__init__(order, basis_type=basis_type,
                         quadrature_degree=quadrature_degree,
                         quadrature_scheme=quadrature_scheme)


def create_time_quadrature(degree, scheme="default"):
    if scheme == "radau":
        num_points = degree + 1
        return RadauQuadratureLineRule(ufc_line, num_points)
    else:
        return create_quadrature(ufc_line, degree, scheme=scheme)
