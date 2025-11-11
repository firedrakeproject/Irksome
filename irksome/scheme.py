from FIAT import ufc_simplex, create_quadrature
from FIAT.quadrature import RadauQuadratureLineRule, GaussLobattoLegendreQuadratureLineRule


ufc_line = ufc_simplex(1)


class GalerkinScheme:
    """
    Base class for describing Galerkin-in-time methods in lieu of
    a Butcher tableau.

    :arg order: An integer indicating the order of the method
    :kwarg basis_type: A string (or tuple of strings) indicating the finite
        element family (either `'Lagrange'` or `'Bernstein'`) or the
        Lagrange variant (either `'equispaced'`, `'spectral'`, `'chebyshev'`,
        or `'integral'`) for the test/trial spaces.
        Defaults to equispaced Lagrange elements.
    :kwarg quadrature_degree: An integer indicating the degree of the
        quadrature to be use in time. Defaults to the sum
        of the degrees of the trial and test spaces.
    :kwarg quadrature_scheme: A string indicating the quadrature scheme
        to be used in time. Defaults to Gauss-Legendre.
    """
    def __init__(self, order,
                 basis_type=None,
                 quadrature_degree=None,
                 quadrature_scheme=None):
        self.order = order
        self.basis_type = basis_type
        self.quadrature_degree = quadrature_degree
        self.quadrature_scheme = quadrature_scheme


class DiscontinuousGalerkinScheme(GalerkinScheme):
    """Class for describing DG-in-time methods"""
    def __init__(self, order,
                 basis_type=None,
                 quadrature_degree=None,
                 quadrature_scheme=None):
        assert order >= 0, f"{type(self).__name__} must have order >= 0"
        super().__init__(order, basis_type=basis_type,
                         quadrature_degree=quadrature_degree,
                         quadrature_scheme=quadrature_scheme)


class ContinuousPetrovGalerkinScheme(GalerkinScheme):
    """Class for describing cPG-in-time methods"""
    def __init__(self, order,
                 basis_type=None,
                 quadrature_degree=None,
                 quadrature_scheme=None):
        assert order >= 1, f"{type(self).__name__} must have order >= 1"
        super().__init__(order, basis_type=basis_type,
                         quadrature_degree=quadrature_degree,
                         quadrature_scheme=quadrature_scheme)


class GalerkinCollocationScheme(ContinuousPetrovGalerkinScheme):
    """Class for describing collocation cPG-in-time methods"""
    def __init__(self, order,
                 stage_type="value",
                 quadrature_degree=None,
                 quadrature_scheme=None):
        assert order >= 1, f"{type(self).__name__} must have order >= 1"
        if quadrature_scheme is None:
            test_type = "spectral"
        elif quadrature_scheme in {"radau", "lobatto"}:
            test_type = quadrature_scheme
            if quadrature_degree is None and quadrature_scheme == "lobatto":
                # Default to under-integration
                quadrature_degree = 2*order-2
        else:
            raise ValueError(f"Unsupported quadrature scheme {quadrature_scheme}.")

        if stage_type not in {"deriv", "value"}:
            raise ValueError(f"Unsupported stage type {stage_type}.")

        trial_type = stage_type
        basis_type = (trial_type, test_type)
        super().__init__(order, basis_type=basis_type,
                         quadrature_degree=quadrature_degree,
                         quadrature_scheme=quadrature_scheme)


def create_time_quadrature(degree, scheme=None):
    if scheme == "radau":
        num_points = degree // 2 + 1
        return RadauQuadratureLineRule(ufc_line, num_points)
    elif scheme == "lobatto":
        num_points = (degree + 1) // 2 + 1
        return GaussLobattoLegendreQuadratureLineRule(ufc_line, num_points)
    else:
        return create_quadrature(ufc_line, degree, scheme=scheme or "default")
