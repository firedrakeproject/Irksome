from firedrake import assemble, project, replace
from firedrake.__future__ import interpolate


def get_sub(u, indices):
    for i in indices:
        u = u.sub(i)
    return u


def bc2space(bc, V):
    return get_sub(V, bc._indices)


def stage2spaces4bc(bc, V, Vbig, i):
    """used to figure out how to apply Dirichlet BC to each stage"""
    num_fields = len(V)
    sub = 0 if num_fields == 1 else bc.function_space_index()
    comp = bc.function_space().component

    Vbigi = Vbig[sub+num_fields*i]
    if comp is not None:  # check for sub-piece of vector-valued
        Vbigi = Vbigi.sub(comp)

    Vsp = bc2space(bc, V)
    return Vsp, Vbigi


def gstuff(V, g):
    if g == 0:
        gdat = g
        gmethod = lambda *args, **kargs: None
    else:
        try:
            gdat = assemble(interpolate(g, V))
            gmethod = lambda gd, gc: gd.interpolate(gc)
        except (NotImplementedError, AttributeError):
            gdat = project(g, V)
            gmethod = lambda gd, gc: gd.project(gc)
    return gdat, gmethod


class BCStageData(object):
    def __init__(self, V, gcur, u0, u0_mult, i, t, dt):
        if V.component is None:  # V is not a bit of a VFS
            if V.index is None:  # not part of MFS, either
                indices = ()
            else:  # part of MFS
                indices = (V.index,)
        else:  # bottommost space is bit of VFS
            if V.parent.index is None:  # but not part of a MFS
                indices = (V.component,)
            else:   # V is a bit of a VFS inside an MFS
                indices = (V.parent.index, V.component)

        if gcur == 0:  # special case DirichletBC(V, 0, ...), do nothing
            gdat = gcur
        else:
            gdat = gcur - u0_mult[i] * get_sub(u0, indices)

        # DirichletBC handles the update
        # an external update callback is not necessary
        gmethod = lambda *args, **kwargs: None
        self.gstuff = (gdat, gcur, gmethod)


class EmbeddedBCData(object):
    def __init__(self, bc, t, dt, num_fields, num_stages, btilde, V, ws, u0):
        Vsp = bc2space(bc, V)
        gorig = bc._original_arg
        if gorig == 0:  # special case DirichletBC(V, 0, ...), do nothing
            gdat = gorig
            gcur = gorig
        else:
            gcur = replace(gorig, {t: t+dt})
            num_fields = len(V)
            sub = 0 if num_fields == 1 else bc.function_space_index()
            comp = bc.function_space().component
            if comp is None:  # check for sub-piece of vector-valued
                for j in range(num_stages):
                    gcur -= dt*btilde[j]*ws[num_fields*j+sub]
            else:
                for j in range(num_stages):
                    gcur -= dt*btilde[j]*ws[num_fields*j+sub].sub(comp)

            gdat = gcur - bc2space(bc, u0)

        # DirichletBC handles the update
        # an external update callback is not necessary
        gmethod = lambda *args, **kwargs: None
        self.gstuff = (gdat, gcur, gmethod, Vsp)
