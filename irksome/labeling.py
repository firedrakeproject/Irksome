from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl import BaseForm, Form, FormSum
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


def has_quad_labels(term):
    return any(isinstance(label, TimeQuadratureRule) for label in term.labels)


def apply_time_quadrature_labels(form, degree_estimator, scheme=None):
    """
    Estimates the polynomial degree in time for each integral in the given form and labels
    each term with a quadrature rule to be used for time integration.

    :arg form: a :class:`BaseForm` or a partially labelled :class:`LabelledForm`.
    :arg degree_estimator: a :class:`TimeDegreeEstimator` instance.
    :kwarg scheme: a string with the quadrature scheme.

    :returns: a :class:`LabelledForm` labelled by :class:`TimeQuaradratureRule` instances.
    """

    # Split labelled and unlabelled parts
    if isinstance(form, LabelledForm):
        F = form.label_map(has_quad_labels, map_if_true=keep, map_if_false=drop)
        form = as_form(form.label_map(has_quad_labels, map_if_true=drop, map_if_false=keep))
    elif isinstance(form, BaseForm):
        F = LabelledForm()
    else:
        raise ValueError(f"Expecting a BaseForm or a LabelledForm, not {type(form).__name__}")

    if not isinstance(form, Form) and isinstance(form, BaseForm):
        # Label the non-integral components
        if isinstance(form, FormSum):
            ws = form.weights()
            fs = form.components()
            form = sum((w*f for f, w in zip(fs, ws) if isinstance(f, Form)), Form([]))
            base_form = FormSum(*((f, w) for f, w in zip(fs, ws) if not isinstance(f, Form)))
        else:
            base_form = form
            form = Form([])

        degree = degree_estimator(apply_algebra_lowering(base_form))
        label = TimeQuadratureLabel(degree, scheme=scheme)
        F += label(base_form)

    if not isinstance(form, Form):
        raise NotImplementedError(f"Expecting a Form, not {type(form).__name__}")

    # Need to preprocess Inverse and Determinant
    form = apply_algebra_lowering(form)

    # Group integrals by degree
    terms = defaultdict(list)
    for it in form.integrals():
        terms[degree_estimator(it)].append(it)

    for degree, integrals in terms.items():
        label = TimeQuadratureLabel(degree, scheme=scheme)
        F += label(Form(integrals))
    return F


def split_quadrature(F, degree_estimator=None, Qdefault=None):
    """Splits a :class:`LabelledForm` into the terms to be integrated in time by the
    different :class:`TimeQuadratureRule` objects used as labels.

    :arg F: a :class:`Form` or a :class:`LabelledForm`.
    :kwarg degree_estimator: a :class:`TimeDegreeEstimator` instance.
    :kwarg Qdefault: the :class:`TimeQuadratureRule` to be applied on unlabelled terms.
        Alternatively, a string indicating the quadrature scheme,
        in which case the degree is automatically estimated for each unlabelled term.

    :returns: a `dict` mapping unique :class:`TimeQuadratureRule` objects to the :class:`Form` to be integrated in time.
    """
    do_estimate_degrees = Qdefault is None or isinstance(Qdefault, str)
    if do_estimate_degrees:
        scheme = Qdefault
        F = apply_time_quadrature_labels(F, degree_estimator, scheme=scheme)

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
    Fdefault = as_form(F.label_map(has_quad_labels, map_if_true=drop, map_if_false=keep))
    if do_estimate_degrees:
        # every term must have been labelled at this point
        assert Fdefault.empty()
    else:
        splitting[Qdefault] = Fdefault
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
