from firedrake.fml import Label, keep, drop, LabelledForm
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
    if not isinstance(F, LabelledForm):
        return {Qdefault: F}

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


def split_explicit(F):
    if not isinstance(F, LabelledForm):
        return (F, None)
    exp_part = F.label_map(lambda t: t.has_label(explicit),
                           map_if_true=keep,
                           map_if_false=drop)

    imp_part = F.label_map(lambda t: t.labels == {},
                           map_if_true=keep, map_if_false=drop)

    return imp_part.form, exp_part.form


def as_form(form):
    if isinstance(form, LabelledForm):
        form = form.form
    return form
