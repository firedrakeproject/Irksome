from firedrake import replace


def get_sub(u, indices):
    for i in indices:
        u = u.sub(i)
    return u


def bc2space(bc, V):
    return get_sub(V, bc._indices)


def stage2spaces4bc(bc, Vbig, i):
    """used to figure out how to apply Dirichlet BC to each stage"""
    V = bc.function_space()
    num_fields = len(V)
    sub = 0 if num_fields == 1 else bc.function_space_index()
    comp = V.component

    Vbigi = Vbig[sub+num_fields*i]
    if comp is not None:  # check for sub-piece of vector-valued
        Vbigi = Vbigi.sub(comp)

    return Vbigi


def BCStageData(V, gcur, u0, u0_mult, i, t, dt):
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
    return gdat


def EmbeddedBCData(bc, t, dt, num_fields, num_stages, btilde, V, ws, u0):
    gorig = bc._original_arg
    if gorig == 0:  # special case DirichletBC(V, 0, ...), do nothing
        gdat = gorig
    else:
        gcur = replace(gorig, {t: t+dt})
        for j in range(num_stages):
            wj = stage2spaces4bc(bc, ws, j)
            gcur -= (dt * btilde[j]) * wj
        gdat = gcur - bc2space(bc, u0)
    return gdat
