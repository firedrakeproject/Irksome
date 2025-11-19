from firedrake.fml import Label, keep, drop, LabelledForm
from .scheme import create_time_quadrature
from ufl.form import Form
from ufl.measure import Measure
import numpy as np

explicit = Label("explicit")


class TimeQuadratureLabel(Label):
    """If the constructor gets one argument, it's an integer for the
    order of the quadrature rule.
    If there are two arguments, assume they are the points and weights."""
    def __init__(self, *args, scheme="default"):
        if len(args) == 1:
            Q = create_time_quadrature(args[0], scheme=scheme)
            x, w = Q.get_points(), Q.get_weights()
        elif len(args) == 2:
            x, w = args
        else:
            raise ValueError("Illegal input to TimeQuadratureLabel")
        super().__init__(TimeQuadratureRule(x, w))


class TimeQuadratureRule:
    def __init__(self, x, w):
        self.x = x
        self.w = w

    def get_points(self):
        return np.asarray(self.x)

    def get_weights(self):
        return np.asarray(self.w)


def split_quadrature(F, Qdefault=None):
    """Split a form into subforms grouped by time quadrature rule.

    Supports two mechanisms:
      1) firedrake.fml labels using TimeQuadratureLabel/TimeQuadratureRule
      2) UFL integral metadata containing Irksome keys
         ("quadrature_degree_time" and optionally "quadrature_scheme_time").

    If neither labelling nor metadata overrides are present, returns
    a single entry mapping Qdefault -> F.
    """
    # Case 1: LabelledForm path (existing behaviour)
    if isinstance(F, LabelledForm):
        quad_labels = set()
        for term in F.terms:
            cur_labels = [label for label in term.labels if isinstance(label, TimeQuadratureRule)]
            if len(cur_labels) == 1:
                quad_labels.update(cur_labels)
            elif len(cur_labels) > 1:
                raise ValueError("Multiple quadrature labels on one term.")

        splitting = {Q: F.label_map(lambda t: Q in t.labels, map_if_true=keep, map_if_false=drop)
                     for Q in quad_labels}
        splitting[Qdefault] = F.label_map(lambda t: len(quad_labels.intersection(t.labels)) > 0,
                                          map_if_true=drop, map_if_false=keep)
        for Q in list(splitting):
            try:
                splitting[Q] = splitting[Q].form
            except TypeError:
                splitting.pop(Q)
        return splitting

    # Case 2: Plain UFL form with per-integral metadata overrides
    # See if I can recover integral; it not, return default
    try:
        integrals = F.integrals()
    except Exception:
        return {Qdefault: F}
    
    # Scan for Irksome metadata; if none present, return default
    IRK_DEG = "quadrature_degree_override"
    IRK_SCH = "quadrature_scheme_override"
    has_override = any(
        (IRK_DEG in (I.metadata() or {}) or IRK_SCH in (I.metadata() or {}))
        for I in integrals
        )
    if not has_override:
        return {Qdefault: F}

    # Since we got here, build groups keyed by (degree, scheme) tuples
    groups = {}
    default_ints = []
    # For each integral...
    for I in integrals:
        # ...get the metadata...
        md = I.metadata() or {}
        deg = md.get(IRK_DEG, None)
        sch = md.get(IRK_SCH, None)
        if deg is None:
            # ...if no quadrature override is specified, add to default...
            default_ints.append(I)
        else:
            # ...and otherwise, record in groups
            sch = sch if sch is not None else "default"
            key = (int(deg), str(sch))
            groups.setdefault(key, []).append(I)

    # Now, assemble into a dictionary as required using create_time_quadrature
    result = {}
    for (deg, sch), ints in groups.items():
        Q = create_time_quadrature(deg, scheme=sch)
        result[Q] = Form(ints)
    if default_ints:
        result[Qdefault] = Form(default_ints)

    return result


def split_explicit(F):
    if not isinstance(F, LabelledForm):
        return (F, None)
    exp_part = F.label_map(lambda t: t.has_label(explicit),
                           map_if_true=keep,
                           map_if_false=drop)

    imp_part = F.label_map(lambda t: t.labels == {},
                           map_if_true=keep, map_if_false=drop)

    return imp_part.form, exp_part.form


class MeasureOverride(Measure):
    """Thin wrappers around UFL Measures that allow users to tag
    individual integrals with Irksome-specific overrides for
    time quadrature used by Galerkin-in-time discretisations.

    Usage example:
    F = inner(Dt(u), v) * dx_override(time_degree_override=5) + inner(u, v) * dx

    Here, only the first term will be integrated in time with a rule of
    degree 5; the other terms will use the scheme defaults.
    """
    def __call__(
        self,
        subdomain_id=None,
        metadata=None,
        domain=None,
        subdomain_data=None,
        degree=None,
        scheme=None,
        *,
        time_degree_override=None,
        time_scheme_override=None,
    ):
        """Reconfigure measure with (optional) time quadrature overrides.

        The optional keyword-only arguments time_degree_override and time_scheme_override
        are stored in metadata keys understood by Irksome's Galerkin-in-time
        machinery in split_quadrature().
        """
        # Inject time overrides into metadata
        if time_degree_override is None and time_scheme_override is not None:
            raise ValueError(
                "Time quadrature override requires specification of time_degree_override."
            )
        if time_degree_override is not None or time_scheme_override is not None:
            metadata = {} if metadata is None else metadata.copy()
            if time_degree_override is not None:
                metadata["quadrature_degree_override"] = time_degree_override
            if time_scheme_override is not None:
                metadata["quadrature_scheme_override"] = time_scheme_override

        # Inject spatial (degree, scheme) into metadata if provided, mirroring
        # ufl.measure.Measure.__call__ semantics.
        if (degree, scheme) != (None, None):
            metadata = {} if metadata is None else metadata.copy()
            if degree is not None:
                metadata["quadrature_degree"] = degree
            if scheme is not None:
                metadata["quadrature_rule"] = scheme

        # Support dx(domain) style: if first positional looks like a domain, treat accordingly
        if subdomain_id is not None and hasattr(subdomain_id, "ufl_domain"):
            if domain is not None:
                raise ValueError(
                    "Ambiguous: setting domain both as keyword argument and first argument."
                )
            subdomain_id, domain = "everywhere", subdomain_id

        # Without args, return everywhere
        if all(x is None for x in (subdomain_id, metadata, domain, subdomain_data, degree, scheme)) and (
            time_degree_override is None and time_scheme_override is None
        ):
            return self.reconstruct(subdomain_id="everywhere")

        # Construct new Measure
        return Measure(
            self.integral_type(),
            domain=domain or self.ufl_domain(),
            subdomain_id=subdomain_id if subdomain_id is not None else self.subdomain_id(),
            metadata=metadata if metadata is not None else self.metadata(),
            subdomain_data=subdomain_data if subdomain_data is not None else self.subdomain_data(),
        )


# Convenience instances mirroring Firedrake/UFL defaults
dx_override = MeasureOverride("cell")
ds_override = MeasureOverride("exterior_facet")
dS_override = MeasureOverride("interior_facet")
dr_override = MeasureOverride("ridge")
dP_override = MeasureOverride("vertex")
dc_override = MeasureOverride("custom")
dC_override = MeasureOverride("cutcell")
dI_override = MeasureOverride("interface")
dO_override = MeasureOverride("overlap")
ds_b_override = MeasureOverride("exterior_facet_bottom")
ds_t_override = MeasureOverride("exterior_facet_top")
ds_v_override = MeasureOverride("exterior_facet_vert")
dS_h_override = MeasureOverride("interior_facet_horiz")
dS_v_override = MeasureOverride("interior_facet_vert")

