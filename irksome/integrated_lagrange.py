from FIAT import finite_element, dual_set, functional
from FIAT.reference_element import LINE
from FIAT.barycentric_interpolation import LagrangePolynomialSet, get_lagrange_points


class IntegratedLagrangeDualSet(dual_set.DualSet):
    """The dual basis for 1D Integrated Lagrange elements."""
    def __init__(self, ref_el, points):
        entity_ids = {0: {0: [0], 1: []},
                      1: {0: list(range(1, len(points)+1))}}

        dpts = points
        pt, = ref_el.make_points(0, 0, 1)
        self.points = [pt, *points]

        nodes = []
        nodes.append(functional.PointEvaluation(ref_el, pt))
        nodes.extend(functional.PointDerivative(ref_el, pt, (1,)) for pt in dpts)
        super().__init__(nodes, ref_el, entity_ids)


class IntegratedLagrange(finite_element.CiarletElement):
    """1D element with derivate DOFs on quadrature points."""
    def __init__(self, element):
        ref_el = element.get_reference_element()
        if ref_el.shape != LINE:
            raise ValueError(f"{type(self).__name__} only defined in one dimension.")

        degree = element.degree() + 1
        points = get_lagrange_points(element.dual_basis())
        dual = IntegratedLagrangeDualSet(ref_el, points)
        poly_set = LagrangePolynomialSet(ref_el, dual.points)
        super().__init__(poly_set, dual, degree)
