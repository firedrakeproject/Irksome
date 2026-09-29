from ufl.classes import Label, Variable


# A :class:`ufl.Label` to mark nodes that are only evaluated at the start of
# the timestep.
lag_label = Label()


def lag(expr):
    """Mark a sub-expression to be evaluated only at the start of the
    timestep during the implicit solve."""
    return Variable(expr, lag_label)
