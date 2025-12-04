from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl import Form
from firedrake.fml import Label, keep, drop, LabelledForm
from collections import defaultdict
from .scheme import create_time_quadrature
from .estimate_degrees import TimeDegreeEstimator
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


def apply_time_quadrature_labels(form, test_degree, trial_degree, t=None, timedep_coeffs=None):
    """
    Estimates the polynomial degree in time for each integral in the given form and labels
    each term with a quadrature rule to be used for time integration.

    :arg form: a :class:`Form` or a partially labelled :class:`LabelledForm`.
    :arg test_degree: the temporal polynomial degree of the test space.
    :arg trial_degree: the temporal polynomial degree of the trial space.
    :kwarg t: the time variable as a :class:`Constant` or :class:`Function` in the Real space.
    :kwarg timedep_coeffs: a list of :class:`Function` that depend on time.

    :returns: a :class:`LabelledForm` labelled by :class:`TimeQuaradratureRule` instances.
    """
    if isinstance(form, Form):
        remainder = None
    elif isinstance(form, LabelledForm):
        remainder = form.label_map(has_quad_labels, map_if_true=keep, map_if_false=drop)
        form = as_form(form.label_map(has_quad_labels, map_if_true=drop, map_if_false=keep))
    else:
        raise ValueError(f"Expecting a Form or LabelledForm, not {type(form).__name__}")
    if form.empty():
        return remainder
    # Need to preprocess Inverse and Determinant
    form = apply_algebra_lowering(form)

    # Group integrals by degree
    de = TimeDegreeEstimator(test_degree, trial_degree, t=t, timedep_coeffs=timedep_coeffs)
    terms = defaultdict(list)
    for it in form.integrals():
        terms[de(it.integrand())].append(it)

    F = remainder
    for deg, its in terms.items():
        label = TimeQuadratureLabel(deg)
        term = label(Form(its))
        if F is None:
            F = term
        else:
            F += term
    return F


def split_quadrature(F, test_degree, trial_degree, t=None, timedep_coeffs=None, Qdefault="auto"):
    """Splits a :class:`LabelledForm` into the terms to be integrated in time by the
    different :class:`TimeQuadratureRule` objects used as labels.

    :arg F: a :class:`Form` or a :class:`LabelledForm`.
    :arg test_degree: the temporal polynomial degree of the test space.
    :arg trial_degree: the temporal polynomial degree of the trial space.
    :kwarg t: the time variable as a :class:`Constant` or :class:`Function` in the Real space.
    :kwarg timedep_coeffs: a list of :class:`Function` that depend on time.
    :kwarg Qdefault: the :class:`TimeQuadratureRule` to be applied on unlabelled terms.

    :returns: a `dict` mapping unique :class:`TimeQuadratureRule` objects to the :class:`Form` to be integrated in time.
    """
    if Qdefault == "auto":
        F = apply_time_quadrature_labels(F, test_degree, trial_degree, t=t, timedep_coeffs=timedep_coeffs)

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
    if Qdefault == "auto":
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
