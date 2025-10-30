class GalerkinScheme:
    """
    Base class for describing Galerkin-in-time methods in lieu of
    a Butcher tableau.

    :arg order: An integer indicating the order of the method
    :kwarg basis_type: A string indicating the finite element family (either
           `'Lagrange'` or `'Bernstein'`) or the Lagrange variant for the
           test/trial spaces. Defaults to equispaced Lagrange elements.
    :kwarg quadrature: A :class:`FIAT.QuadratureRule` indicating the quadrature
            to be used in time, defaulting to GL with order points
    """
    def __init__(self, order, basis_type, quadrature):
        self.order = order
        self.basis_type = basis_type
        self.quadrature = quadrature


class DiscontinuousGalerkinScheme(GalerkinScheme):
    """Class for describing DG-in-time methods"""
    def __init__(self, order, basis_type=None, quadrature=None):
        assert order >= 0, "DG must have order >= 1"
        super().__init__(order, basis_type, quadrature)


class ContinuousPetrovGalerkinScheme(GalerkinScheme):
    """Class for describing cPG-in-time methods"""
    def __init__(self, order, basis_type=None, quadrature=None):
        assert order >= 1, "CPG must have order >= 1"
        super().__init__(order, basis_type, quadrature)
