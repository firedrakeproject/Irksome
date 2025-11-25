from ufl import Form
from firedrake.fml import Label, keep, drop, LabelledForm
from collections import defaultdict
from .scheme import create_time_quadrature
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
    """Splits a :class:`LabelledForm` into the terms to be integrated in time by the
    different :class:`TimeQuadratureRule` objects used as labels.

    :arg F: a :class:`LabelledForm`.
    :kwarg Qdefault: the :class:`TimeQuadratureRule` to be applied on unlabelled terms.

    :returns: a `dict` mapping unique :class:`TimeQuadratureRule` objects to the :class:`Form` to be integrated in time.
    """
    if not isinstance(F, LabelledForm):
        return {Qdefault: F}

    quad_labels = set()
    for term in F.terms:
        cur_labels = [label for label in term.labels if isinstance(label, TimeQuadratureRule)]
        if len(cur_labels) == 1:
            quad_labels.update(cur_labels)
        elif len(cur_labels) > 1:
            raise ValueError("Multiple quadrature labels on one term.")

    splitting = {}
    splitting[Qdefault] = F.label_map(lambda t: len(quad_labels.intersection(t.labels)) == 0,
                                      map_if_true=keep, map_if_false=drop)
    for Q in quad_labels:
        splitting[Q] = F.label_map(lambda t: Q in t.labels, map_if_true=keep, map_if_false=drop)

    # collapse TimeQuadratureRules based on numerical equality
    rule_equals = lambda Q1, Q2: (type(Q1) == type(Q2)
                                  and np.array_equal(Q1.get_points(), Q2.get_points())
                                  and np.array_equal(Q1.get_weights(), Q2.get_weights()))

    forms = defaultdict(lambda: Form([]))
    for Q in sorted(splitting, key=lambda Q: tuple(Q.get_points()), reverse=True):
        form = as_form(splitting[Q])
        if not form.empty():
            Q_unique = next((Qk for Qk in forms if rule_equals(Q, Qk)), Q)
            forms[Q_unique] += form
    return forms


def split_explicit(F):
    if not isinstance(F, LabelledForm):
        return (F, None)
    exp_part = F.label_map(lambda t: t.has_label(explicit),
                           map_if_true=keep,
                           map_if_false=drop)

    imp_part = F.label_map(lambda t: t.labels == {},
                           map_if_true=keep, map_if_false=drop)

    return as_form(imp_part), as_form(exp_part)


def as_form(form):
    """Extracts the :class:`Form` from a :class:`LabelledForm`."""
    if isinstance(form, LabelledForm):
        form = Form([]) if len(form) == 0 else form.form
    return form
