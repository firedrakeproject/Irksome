from FIAT import ufc_simplex, create_quadrature
from FIAT.quadrature import RadauQuadratureLineRule, GaussLobattoLegendreQuadratureLineRule


ufc_line = ufc_simplex(1)


class GalerkinScheme:
    """
    Base class for describing Galerkin-in-time methods in lieu of
    a Butcher tableau.

    :arg order: An integer indicating the order of the method
    :kwarg basis_type: A string (or tuple of strings) indicating the finite
        element family (either ``'Lagrange'`` or ``'Bernstein'``) or the
        Lagrange variant (either ``'equispaced'``, ``'spectral'``, ``'chebyshev'``,
        or ``'integral'``) for the test/trial spaces.
        Defaults to equispaced Lagrange elements.
    :kwarg quadrature_degree: An integer or string indicating the degree of the
        quadrature to be used in time. Defaults to the sum
        of the degrees of the trial and test spaces. The string value ``'auto'``
        enables automatic degree estimation for each term in the form.
    :kwarg quadrature_scheme: A string indicating the quadrature scheme
        to be used in time. Defaults to Gauss-Legendre.
    :kwarg max_quadrature_degree: An integer indicating the maximum quadrature
        degree allowed in the automatic degree estimation.
        If ``None``, then the estimated quadrature degree will always be used.
    """
    def __init__(self, order,
                 basis_type=None,
                 quadrature_degree=None,
                 quadrature_scheme=None,
                 max_quadrature_degree=None):
        self.order = order
        self.basis_type = basis_type
        self.quadrature_degree = quadrature_degree
        self.quadrature_scheme = quadrature_scheme
        self.max_quadrature_degree = max_quadrature_degree

    def __repr__(self):
        return f"{type(self).__name__}({self.order}, {self.basis_type})"


class DiscontinuousGalerkinScheme(GalerkinScheme):
    """Class for describing DG-in-time methods

    :arg order: An integer indicating the order of the method
    :kwarg basis_type: A string (or tuple of strings) indicating the finite
        element family (either ``'Lagrange'`` or ``'Bernstein'``) or the
        Lagrange variant (either ``'equispaced'``, ``'spectral'``, ``'chebyshev'``,
        or ``'integral'``) for the test/trial spaces.
        Defaults to equispaced Lagrange elements.
    :kwarg quadrature_degree: An integer or string indicating the degree of the
        quadrature to be used in time. Defaults to the sum
        of the degrees of the trial and test spaces. The string value ``'auto'``
        enables automatic degree estimation for each term in the form.
    :kwarg quadrature_scheme: A string indicating the quadrature scheme
        to be used in time. Defaults to Gauss-Legendre.
    :kwarg max_quadrature_degree: An integer indicating the maximum quadrature
        degree allowed in the automatic degree estimation.
        If ``None``, then the estimated quadrature degree will always be used.
    :kwarg deriv_type: A string indicating how to integrate terms with time derivatives.
        Valid values are:
        - `'weak'`: Time derivatives act on the test function (integrating by parts once).
        - `'strong'`: Time derivatives act on the unknown (integrating by parts twice).
    """
    def __init__(self, order,
                 basis_type=None,
                 quadrature_degree=None,
                 quadrature_scheme=None,
                 max_quadrature_degree=None,
                 deriv_type="strong"):
        if order < 0:
            raise ValueError(f"{type(self).__name__} must have order >= 0")
        if deriv_type not in {'weak', 'strong'}:
            raise ValueError("deriv_type must be either 'weak' or 'strong'.")
        self.deriv_type = deriv_type
        self.num_stages = order + 1
        super().__init__(order, basis_type=basis_type,
                         quadrature_degree=quadrature_degree,
                         quadrature_scheme=quadrature_scheme,
                         max_quadrature_degree=max_quadrature_degree)


class ContinuousPetrovGalerkinScheme(GalerkinScheme):
    """Class for describing cPG-in-time methods"""
    def __init__(self, order,
                 basis_type=None,
                 quadrature_degree=None,
                 quadrature_scheme=None,
                 max_quadrature_degree=None):
        if order < 1:
            raise ValueError(f"{type(self).__name__} must have order >= 1")
        self.num_stages = order
        super().__init__(order, basis_type=basis_type,
                         quadrature_degree=quadrature_degree,
                         quadrature_scheme=quadrature_scheme,
                         max_quadrature_degree=max_quadrature_degree)


class GalerkinCollocationScheme(ContinuousPetrovGalerkinScheme):
    """Class for describing collocation cPG-in-time methods"""
    def __init__(self, order,
                 stage_type="deriv",
                 quadrature_degree=None,
                 quadrature_scheme=None,
                 max_quadrature_degree=None):
        if order < 1:
            raise ValueError(f"{type(self).__name__} must have order >= 1")
        if quadrature_scheme is None:
            test_type = "spectral"
        elif quadrature_scheme in {"radau", "lobatto"}:
            test_type = quadrature_scheme
        else:
            raise ValueError(f"Unsupported quadrature scheme {quadrature_scheme}.")

        if quadrature_degree is None:
            # Default to under-integration
            quadrature_degree = collocation_quadrature_degree(order, quadrature_scheme)

        if stage_type not in {"deriv", "value"}:
            raise ValueError(f"Unsupported stage type {stage_type}.")

        trial_type = stage_type
        basis_type = (trial_type, test_type)
        self.stage_type = stage_type
        super().__init__(order, basis_type=basis_type,
                         quadrature_degree=quadrature_degree,
                         quadrature_scheme=quadrature_scheme,
                         max_quadrature_degree=max_quadrature_degree)

    def as_collocation_scheme(self):
        """Return a true collocation scheme with collocated quadrature"""
        return type(self)(self.order, stage_type=self.stage_type)


class DiscontinuousGalerkinCollocationScheme(DiscontinuousGalerkinScheme):
    """Class for describing collocation DG-in-time methods"""
    def __init__(self, order,
                 deriv_type="strong",
                 quadrature_degree=None,
                 quadrature_scheme="radau",
                 max_quadrature_degree=None):
        if order < 0:
            raise ValueError(f"{type(self).__name__} must have order >= 0")
        if quadrature_scheme is None:
            basis_type = "spectral"
        elif quadrature_scheme in {"radau", "lobatto"}:
            basis_type = quadrature_scheme
        else:
            raise ValueError(f"Unsupported quadrature scheme {quadrature_scheme}.")

        if quadrature_degree is None:
            # Default to under-integration
            quadrature_degree = collocation_quadrature_degree(order+1, quadrature_scheme)

        super().__init__(order, basis_type=basis_type,
                         deriv_type=deriv_type,
                         quadrature_degree=quadrature_degree,
                         quadrature_scheme=quadrature_scheme,
                         max_quadrature_degree=max_quadrature_degree)

    def as_collocation_scheme(self):
        """Return a true collocation scheme with collocated quadrature"""
        return type(self)(self.order, deriv_type=self.deriv_type)


def create_time_quadrature(degree, scheme=None):
    """Return a :class:`FIAT.QuadratureRule` on the unit interval.

    :arg degree: The degree of polynomial that the rule should integrate exactly.
    :kwarg scheme: The quadrature scheme. Can be either `'default'` for Gauss-Legendre,
        `'Lobatto'` for Gauss-Lobatto-Legendre, or `'Radau'` for Gauss-Radau.
    """
    if scheme is not None and scheme.lower() == "radau":
        num_points = (degree + 1) // 2 + 1
        return RadauQuadratureLineRule(ufc_line, num_points)
    elif scheme is not None and scheme.lower() == "lobatto":
        num_points = degree // 2 + 2
        return GaussLobattoLegendreQuadratureLineRule(ufc_line, num_points)
    else:
        return create_quadrature(ufc_line, degree, scheme=scheme or "default")


def collocation_quadrature_degree(trial_degree, scheme):
    """Return the quadrature degree for a collocation scheme."""
    if scheme is None:
        scheme = "default"
    scheme = scheme.lower()
    if scheme == "default":
        degree = 2*trial_degree-1
    elif scheme == "radau":
        degree = 2*trial_degree-2
    elif scheme == "lobatto":
        degree = 2*trial_degree-3
    else:
        raise ValueError(f"Unrecognized quadrature scheme {scheme}")
    return degree
