from firedrake.fml import Label, keep, drop, LabelledForm


explicit = Label("explicit")
empty_label = Label("")


class TimeQuadratureLabel(Label):
    def __init__(self, x, w):
        super().__init__(TimeQuadratureRule(x, w))


class TimeQuadratureRule:
    def __init__(self, x, w):
        self.x = x
        self.w = w


def split_explicit(F):
    if not isinstance(F, LabelledForm):
        return (F, None)
    exp_part = F.label_map(lambda t: t.has_label(explicit),
                           map_if_true=keep,
                           map_if_false=drop)

    imp_part = F.label_map(lambda t: t.labels == {},
                           map_if_true=keep, map_if_false=drop)

    return imp_part.form, exp_part.form


def as_labelled_form(F):
    if not isinstance(F, LabelledForm):
        return empty_label(F)
    else:
        return F
