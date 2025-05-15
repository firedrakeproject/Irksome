from firedrake.fml import Label, keep, drop, LabelledForm
from ufl.form import Form


explicit = Label("explicit")


def split_explicit(F):
    if isinstance(F, Form):
        return (F, None)
    assert isinstance(F, LabelledForm)
    exp_part = F.label_map(lambda t: t.has_label(explicit),
                           map_if_true=keep,
                           map_if_false=drop)

    imp_part = F.label_map(lambda t: t.labels == {},
                           map_if_true=keep, map_if_false=drop)

    return imp_part.form, exp_part.form
