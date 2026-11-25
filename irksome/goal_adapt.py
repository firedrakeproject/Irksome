"""The functions behind dwr_time_demo.py (goal-oriented adaptivity in TIME only).

# DWR only decides which time slabs to bisect; the spatial mesh is fixed.
    solve_primal                one stepper advanced slab by slab over the time nodes
    dwr_spacetime_adaptivity    SOLVE (adjoint) -> ESTIMATE -> MARK -> REFINE (bisect slabs)
    solve_adjoint_slabs         adjoint, backwards, one small Irksome stepper per slab

    primal_residual             rho(u_h)(e_z): the global estimate
    adjoint_residual            rho*(u_h, z_h)(e_u), outgoing temporal trace
    recover_densities           bubble-cone recovery of the vol / sf / tf / xt densities
    bubble_cone_indicators      eta_{K,n}: recovered densities paired with e_z and e_u
    mark, refine_time           slabs ranked by sum_K |eta_{K,n}|, bisection of the marked slabs
    evaluate_goal, reference_goal

Discretisation: primal cG(p) x dG(q) from the user's stepper;
z+, u+ in cG(p) x dG(q+2) (enriched in time only, since only the time error is estimated).
"""
import time as clock
import numpy
from finat.ufl import BrokenElement, FiniteElement
from firedrake import *
from irksome import DiscontinuousGalerkinScheme, Dt, TimeStepper
from irksome.ufl.manipulation import remove_time_derivatives, split_time_derivative_terms
import ufl
from ufl import replace
from ufl.algorithms import expand_derivatives, extract_coefficients

__all__ = ["solve_primal", "dwr_spacetime_adaptivity", "reference_goal", "print_estimates"]

LU = {"ksp_type": "preonly", "pc_type": "lu"}
RECOVERY_SOLVER = {"mat_type": "aij", "ksp_type": "cg", "pc_type": "jacobi",
                   "ksp_rtol": 1e-11, "ksp_atol": 1e-13, "ksp_max_it": 1000}


def _end_values(el):
    """Values of the dG time basis at the right end s = 1 of the reference slab."""
    # phi_i(1), in order to compute U(t_{n+1}^-) = sum_i phi_i(1) U_i
    return numpy.asarray(el.tabulate(0, [(1.0,)])[(0,)])[:, 0]


def solve_primal(stepper, nodes):
    """Advance stepper over the slabs [t_n, t_{n+1}]; one record (u_in, stages) per slab."""
    primal = []
    for n in range(len(nodes) - 1):
        stepper.t.assign(nodes[n])
        stepper.dt.assign(nodes[n + 1] - nodes[n])
        u_in = stepper.u0.copy(deepcopy=True)
        stepper.advance()
        primal.append({"u_in": u_in,
                       "stages": [s.copy(deepcopy=True) for s in stepper.stages.subfunctions]})
    return primal


def _setup(stepper, goal, terminal_goal):
    u, t = stepper.u0, stepper.t
    v, = stepper.F.arguments()
    V = u.function_space()
    if len(V) > 1:
        raise NotImplementedError("This demo backend handles one scalar field.")
    split = split_time_derivative_terms(stepper.F, t=t, timedep_coeffs=(u,))
    m_form = remove_time_derivatives(split.time)
    F_x = split.remainder       # F(u, v; t) without Dt
    z = Function(V, name="z_placeholder")
    # adjoint operator  F'(u)(v, z): derivative of F(u, z) in u, in the direction v
    A = expand_derivatives(derivative(action(F_x, z), u, v))
    dj = None if goal is None else expand_derivatives(derivative(goal, u, v))
    dJT = None if terminal_goal is None else expand_derivatives(derivative(terminal_goal, u, v))
    # nonlinear adjoint
    depends_on_u = any(u in extract_coefficients(f) for f in (A, dj) if f is not None)
    nonlinear = depends_on_u or (dJT is not None and u in extract_coefficients(dJT))
    return dict(u=u, v=v, t=t, V=V, z=z, m=m_form, F_x=F_x, A=A, dj=dj,
                dJT=dJT, depends_on_u=depends_on_u, nonlinear=nonlinear,
                goal=goal, terminal_goal=terminal_goal, F=stepper.F,
                el=stepper.el, q=stepper.order, primal_bcs=stepper.orig_bcs,
                bcs=[DirichletBC(V, 0, bc.sub_domain) for bc in stepper.orig_bcs])


def _polynomial_in(el, s):
    """UFL expressions of the dG time basis functions at s (any UFL scalar)."""
    n = el.space_dimension()
    nodes = numpy.linspace(0.1, 0.9, n) if n > 1 else numpy.array([0.5])
    vals = numpy.asarray(el.tabulate(0, [(x,) for x in nodes])[(0,)])
    coeffs = numpy.linalg.solve(numpy.vander(nodes, n, increasing=True), vals.T)
    return [sum(float(coeffs[j, i]) * s**j for j in range(n)) for i in range(n)]


# ----------------------------------------------------------------------------------------
# DWR: SOLVE (adjoint) -> ESTIMATE -> MARK -> REFINE
# ----------------------------------------------------------------------------------------
def dwr_spacetime_adaptivity(stepper, nodes, primal, goal, terminal_goal=None, *,
                             theta=0.5, exact_goal=None, estimates=None,
                             npoints=12, time_enrichment=2, estimator="auto",
                             localisation="slab", recovery_degrees=None):
    """One DWR step in time. Returns (the refined time nodes, eta).

    estimator : "primal"     eta = rho(u_h)(z+ - Pi_h z+)                              (2.23)
                "symmetric"  eta = 1/2 rho(u_h)(z+ - z_h) + 1/2 rho*(u_h, z_h)(u+ - u_h)   (2.22)
                "auto"       "symmetric" for a nonlinear F or goal, "primal" otherwise.
    z+, u+ are dG(q + time_enrichment) in time; Pi_h interpolates in time onto dG(q).
    localisation : "slab"         eta_n = the residuals restricted to slab n (a partition of
                                  unity in time); sum_n eta_n = eta exactly; Doerfler on |eta_n|
                   "bubble_cone"  eta_{K,n} from the recovered densities (Section 2.3);
                                  marking with mark()
    recovery_degrees : (cell, facet, time) degrees of the recovered densities, (A3);
                       default (p, p, q + 1). Only used by "bubble_cone".
    """
    P = _setup(stepper, goal, terminal_goal)     # read F, m, A, dj, dJT, etc. from the stepper
    P["q_plus"] = P["q"] + time_enrichment
    N = len(nodes) - 1
    assert len(primal) == N, f"{len(primal)} recorded slabs for {N} slabs"
    U = [{"stages": rec["stages"], "el": P["el"]} for rec in primal]
    U_in = [rec["u_in"] for rec in primal]                                # U_n^- in every slab
    stages = [rec["stages"] for rec in primal]
    J_h = evaluate_goal(P, nodes, stages, npoints)

    row = {"iteration": len(estimates) if estimates is not None else 0, "Nt": N,
           "dofs": P["V"].dim(), "J_h": J_h, "kmin": numpy.diff(nodes).min()}
    if estimator == "auto":
        estimator = "symmetric" if P["nonlinear"] else "primal"
    row["estimator"] = estimator
    start = clock.perf_counter()
    Zp = solve_adjoint_slabs(P, nodes, stages)                            # SOLVE: z+
    row["time_adjoint"] = clock.perf_counter() - start
    # ESTIMATE: eta_{K,n} for every cell K and slab n
    # the global estimate is the residual itself; the indicators eta_{K,n} pair the
    # bubble-cone recovered densities with the same weights (Section 2.3)
    if estimator == "primal":
        e_z = [_difference((Zp[n], True), interpolate_onto=P["el"]) for n in range(N)]
        eta_n = primal_residual(P, nodes, U, U_in, e_z, npoints).sum(axis=1)     # rho_n(e_z)
        eta = float(eta_n.sum())
        if localisation == "bubble_cone":
            eta_Kn = bubble_cone_indicators(P, nodes, U, U_in, e_z, npoints, recovery_degrees)
    else:
        U_plus = solve_enriched_primal(P, nodes, U_in[0], P["q_plus"])        # u+
        Z_h = solve_adjoint_slabs(P, nodes, stages, degree=P["q"])            # z_h
        z_T = terminal_adjoint_state(P, stages[-1])
        e_z = [_difference((Zp[n], True), (Z_h[n], True)) for n in range(N)]
        e_u = [_difference((U_plus[n], False), (U[n], False)) for n in range(N)]
        rho_n = primal_residual(P, nodes, U, U_in, e_z, npoints).sum(axis=1)            # rho_n
        rho_star_n = adjoint_residual(P, nodes, U, Z_h, e_u, npoints, z_T).sum(axis=1)  # rho*_n
        rho, rho_star = float(rho_n.sum()), float(rho_star_n.sum())
        row["eta_rho"], row["eta_rho_star"] = 0.5 * rho, 0.5 * rho_star
        eta = 0.5 * rho + 0.5 * rho_star
        eta_n = 0.5 * rho_n + 0.5 * rho_star_n
        if localisation == "bubble_cone":
            eta_Kn = bubble_cone_indicators(P, nodes, U, U_in, e_z, npoints, recovery_degrees,
                                            Z=Z_h, e_u=e_u, z_T=z_T)
    row["eta"], row["eta_n"] = eta, eta_n
    if localisation == "bubble_cone":
        row["eta_Kn"] = eta_Kn
        row["eta_local"] = float(eta_Kn.sum())                            # (2.64)
        row["gap"] = abs(row["eta_local"] - eta) / abs(eta) if eta != 0 else numpy.nan   # (2.65)
    if exact_goal is not None:
        row["error"] = exact_goal - J_h
        row["Ieff"] = row["eta"] / row["error"] if row["error"] != 0 else numpy.nan
    # MARK: Doerfler on the slabs; "slab" localisation has one indicator per slab
    marked = mark(row["eta_Kn"] if localisation == "bubble_cone" else eta_n[:, None], theta)
    row["marked"] = len(marked)
    if estimates is not None:
        estimates.append(row)
    return refine_time(nodes, marked), row["eta"]                         # REFINE


def _adjoint_problem(P, z, v, tau, T_total, U_expr=None):
    """m(Dt z, v) + F'(U)(v, z) - j'(U)(v) in reverse time tau (t = T_total - tau)."""
    mapping = {P["z"]: z, P["v"]: v, P["t"]: T_total - tau}
    if P["depends_on_u"]:
        mapping[P["u"]] = U_expr
    F = replace(P["m"], {P["u"]: Dt(z), P["v"]: v}) + replace(P["A"], mapping)
    if P["dj"] is not None:
        F -= replace(P["dj"], mapping)
    return F, P["bcs"]


def terminal_adjoint_state(P, U_last):
    """z(T): m(z(T), v) = J_T'(U(T))(v); zero without a terminal goal."""
    V = P["V"]
    zT = Function(V, name="z_T")
    if P["dJT"] is not None:
        phi = _end_values(P["el"])
        UT = Function(V).assign(sum(float(c) * s for c, s in zip(phi, U_last)))
        solve(replace(P["m"], {P["u"]: TrialFunction(V)}) == replace(P["dJT"], {P["u"]: UT}),
              zT, bcs=P["bcs"], solver_parameters=LU)
    return zT


def solve_adjoint_slabs(P, nodes, U, degree=None):
    """Backwards in time, one Irksome dG stepper per slab."""
    T_total = Constant(nodes[0] + nodes[-1])
    N = len(nodes) - 1
    degree = P["q_plus"] if degree is None else degree
    scheme = DiscontinuousGalerkinScheme(degree, quadrature_degree=2 * degree + 2)
    z_next = terminal_adjoint_state(P, U[-1])
    Z = [None] * N
    for n in reversed(range(N)):                 # n = N-1, ..., 0
        tau, dtau = Constant(nodes[0] + nodes[-1] - nodes[n + 1]), Constant(nodes[n + 1] - nodes[n])
        z = Function(P["V"]).assign(z_next)      # initial value of current slab = z(t_{n+1}) computed on next slab
        U_expr = None
        if P["depends_on_u"]:            # linearise at U(t), t = T_total - tau on this slab
            s = (T_total - tau - nodes[n]) / dtau
            U_expr = sum(b * Ui for b, Ui in zip(_polynomial_in(P["el"], s), U[n]))
        F, bcs = _adjoint_problem(P, z, TestFunction(P["V"]), tau, T_total, U_expr)
        adj = TimeStepper(F, scheme, tau, dtau, z, bcs=bcs, solver_parameters=LU)
        adj.advance()
        Z[n] = {"stages": [s.copy(deepcopy=True) for s in adj.stages.subfunctions], "el": adj.el}
        z_next = z                       # z(t_n^+): the start (in tau) of slab n - 1
    return Z


def solve_enriched_primal(P, nodes, u_init, degree):
    V = P["V"]
    w = Function(V).assign(u_init)
    tp, dtp = Constant(nodes[0]), Constant(nodes[1] - nodes[0])
    F = replace(P["F"], {P["u"]: w, P["t"]: tp})
    bcs = [DirichletBC(V, replace(bc.function_arg, {P["t"]: tp})
                       if isinstance(bc.function_arg, ufl.core.expr.Expr) else bc.function_arg,
                       bc.sub_domain) for bc in P["primal_bcs"]]
    stepper = TimeStepper(F, DiscontinuousGalerkinScheme(degree, quadrature_degree=2 * degree + 6),
                          tp, dtp, w, bcs=bcs, solver_parameters=LU)
    U_plus = []
    for n in range(len(nodes) - 1):
        tp.assign(nodes[n])
        dtp.assign(nodes[n + 1] - nodes[n])
        stepper.advance()
        U_plus.append({"stages": [s.copy(deepcopy=True) for s in stepper.stages.subfunctions],
                       "el": stepper.el})
    return U_plus


def _tabulate(rec, s, reverse=False):
    """Time basis of a slab record at the points s of the forward reference slab [0, 1].
    Adjoint records are stored in reverse time: s -> 1 - s."""
    points = [(1.0 - x,) if reverse else (x,) for x in numpy.atleast_1d(s)]
    return numpy.asarray(rec["el"].tabulate(0, points)[(0,)])


def _difference(first, second=None, interpolate_onto=None):
    """The DWR weight on one slab, as (spatial functions, their time coefficients at s)."""
    rec, reverse = first
    if second is not None:
        rec2, reverse2 = second
        return (rec["stages"] + rec2["stages"],
                lambda s: numpy.vstack([_tabulate(rec, s, reverse), -_tabulate(rec2, s, reverse2)]))
    el = interpolate_onto
    time_nodes = [next(iter(node.get_point_dict()))[0] for node in el.dual.nodes]
    at_nodes = _tabulate(rec, time_nodes, reverse)
    low = {"el": el}
    return (rec["stages"], lambda s: _tabulate(rec, s, reverse) - at_nodes @ _tabulate(low, s))


def adjoint_residual(P, nodes, U, Z, e_u, npoints, z_T):
    """Cell-slab values of rho*(u_h, z_h)(e_u)."""
    V = P["V"]
    cell = TestFunction(FunctionSpace(V.mesh(), "DG", 0))       # one value per cell K
    Us, Zs, dZs, Phi, Jump = (Function(V) for _ in range(5))
    ts = Constant(0.0)
    volume = replace(P["m"], {P["u"]: Phi * cell, P["v"]: dZs}) - \
        replace(P["A"], {P["v"]: Phi * cell, P["z"]: Zs, P["u"]: Us, P["t"]: ts})
    if P["dj"] is not None:
        volume += replace(P["dj"], {P["v"]: Phi * cell, P["u"]: Us, P["t"]: ts})
    jump = replace(P["m"], {P["u"]: Phi * cell, P["v"]: Jump})
    s, w = _gauss(npoints)                       # Gauss points and weights on [0, 1]

    def combine(coefficients, functions):
        return sum(float(a) * x for a, x in zip(coefficients, functions))

    N = len(U)
    rho = []
    for n in range(N):
        k = nodes[n + 1] - nodes[n]
        functions, coefficients = e_u[n]
        e, e1 = coefficients(s), coefficients(1.0)[:, 0]
        phiU = _tabulate(U[n], s)
        tabZ = Z[n]["el"].tabulate(1, [(1.0 - x,) for x in s])   # z_h is stored in reverse time
        zeta, dzeta = numpy.asarray(tabZ[(0,)]), -numpy.asarray(tabZ[(1,)]) / k
        values = 0.0
        for j in range(len(s)):
            Us.assign(combine(phiU[:, j], U[n]["stages"]))
            Zs.assign(combine(zeta[:, j], Z[n]["stages"]))
            dZs.assign(combine(dzeta[:, j], Z[n]["stages"]))
            Phi.assign(combine(e[:, j], functions))
            ts.assign(nodes[n] + k * s[j])
            values = values + k * w[j] * assemble(volume).dat.data_ro
        Phi.assign(combine(e1, functions))                       # phi(t_n^-)
        z_next = z_T if n == N - 1 else combine(_tabulate(Z[n + 1], 0.0, True)[:, 0], Z[n + 1]["stages"])
        Jump.assign(z_next - combine(_tabulate(Z[n], 1.0, True)[:, 0], Z[n]["stages"]))
        rho.append(values + assemble(jump).dat.data_ro)
    return numpy.array(rho)


def _gauss(npoints):
    # Gauss-Legendre, [-1, 1] -> [0, 1]
    x, w = numpy.polynomial.legendre.leggauss(npoints)
    return 0.5 * (x + 1.0), 0.5 * w


def primal_residual(P, nodes, U, U_in, e_z, npoints):
    """Cell-slab values of rho(u_h)(e_z):
       rho_{K,n}(phi) = - int_{I_n} [ m(du_h/dt, phi) + F(u_h, phi; t) ]_K dt
                        - m_K(u_h(t_{n-1}^+) - u_h(t_{n-1}^-), phi(t_{n-1}^+))."""
    V = P["V"]
    cell = TestFunction(FunctionSpace(V.mesh(), "DG", 0))       # one value per cell K
    Us, dUs, Ws, Jump = Function(V), Function(V), Function(V), Function(V)
    ts = Constant(0.0)
    volume = replace(P["m"], {P["u"]: dUs, P["v"]: Ws * cell}) + \
        replace(P["F_x"], {P["u"]: Us, P["v"]: Ws * cell, P["t"]: ts})
    jump = replace(P["m"], {P["u"]: Jump, P["v"]: Ws * cell})
    s, w = _gauss(npoints)
    tab = P["el"].tabulate(1, [(x,) for x in s])
    phi, dphi = numpy.asarray(tab[(0,)]), numpy.asarray(tab[(1,)])
    phi0 = numpy.asarray(P["el"].tabulate(0, [(0.0,)])[(0,)])[:, 0]
    eta = []
    for n in range(len(U)):
        k = nodes[n + 1] - nodes[n]
        Un = U[n]["stages"]
        functions, coefficients = e_z[n]
        e, e0 = coefficients(s), coefficients(0.0)[:, 0]
        values = 0.0
        for j in range(len(s)):
            Us.assign(sum(float(a) * x for a, x in zip(phi[:, j], Un)))
            dUs.assign(sum(float(a) / k * x for a, x in zip(dphi[:, j], Un)))
            Ws.assign(sum(float(a) * x for a, x in zip(e[:, j], functions)))
            ts.assign(nodes[n] + k * s[j])
            values = values - k * w[j] * assemble(volume).dat.data_ro
        Jump.assign(sum(float(a) * x for a, x in zip(phi0, Un)) - U_in[n])
        Ws.assign(sum(float(a) * x for a, x in zip(e0, functions)))
        eta.append(values - assemble(jump).dat.data_ro)
    return numpy.array(eta)


def evaluate_goal(P, nodes, U, npoints):
    """J(U) = int_0^T j(U(t), t) dt + J_T(U(T)), Gauss in time on every slab."""
    V = P["V"]
    Us, ts = Function(V), Constant(0.0)
    s, w = _gauss(npoints)
    phi = numpy.asarray(P["el"].tabulate(0, [(x,) for x in s])[(0,)])
    value = 0.0
    if P["goal"] is not None:
        j = replace(P["goal"], {P["u"]: Us, P["t"]: ts})
        for n, Un in enumerate(U):
            k = nodes[n + 1] - nodes[n]
            for jj in range(len(s)):
                Us.assign(sum(float(a) * x for a, x in zip(phi[:, jj], Un)))
                ts.assign(nodes[n] + k * s[jj])
                value += k * w[jj] * assemble(j)
    if P["terminal_goal"] is not None:
        Us.assign(sum(float(a) * x for a, x in zip(_end_values(P["el"]), U[-1])))   # U(T)
        value += assemble(replace(P["terminal_goal"], {P["u"]: Us}))
    return float(value)


# ----------------------------------------------------------------------------------------
# Localisation: bubble-cone recovery of the residual densities
# ----------------------------------------------------------------------------------------
def _lagrange(nodes, x):
    """Values and derivatives at x of the Lagrange polynomials on nodes."""
    values, derivatives = [], []
    for i, xi in enumerate(nodes):
        others = [xj for j, xj in enumerate(nodes) if j != i]
        values.append(numpy.prod([(x - xj) / (xi - xj) for xj in others]))
        derivatives.append(sum(numpy.prod([(x - xj) / (xi - xj) for xj in others[:m] + others[m + 1:]])
                               / (xi - xm) for m, xm in enumerate(others)))
    return values, derivatives


def _combine(coefficients, functions):
    return sum((float(c) * f for c, f in zip(coefficients, functions)), Constant(0.0) * functions[0])


def _at(rec, x, k=None, reverse=False):
    """The slab polynomial of a record at reference time x; with k, its time derivative."""
    if k is None:
        return _combine(_tabulate(rec, x, reverse)[:, 0], rec["stages"])
    point = [(1.0 - x,) if reverse else (x,)]
    derivative = numpy.asarray(rec["el"].tabulate(1, point)[(1,)])[:, 0]
    return _combine((-derivative if reverse else derivative) / k, rec["stages"])


def recover_densities(mesh, t0, k, volume_action, temporal_action, endpoint, npoints, degrees):
    """Recover the vol / sf / tf / xt densities of one linear residual functional on a slab."""
    dim = mesh.topological_dimension
    space_degree, facet_degree, time_degree = degrees
    density_space = FunctionSpace(mesh, "DG", space_degree, variant="integral")
    time_nodes = numpy.linspace(0.0, 1.0, time_degree + 1) if time_degree > 0 else numpy.array([0.5])
    # b_K = prod_{i=1}^{d+1} lambda_i^K
    bubble = Function(FunctionSpace(mesh, "B", dim + 1, variant="integral")).assign(1.0)
    if dim == 1:
        facet_space = FunctionSpace(mesh, "DG", 1, variant="equispaced")
        cone = 1.0
    else:
        # beta_{K,F} = prod_{i in I_F} lambda_i^K
        cone = Function(FunctionSpace(mesh, "FB", dim, variant="integral")).assign(1.0)
        facet_space = FunctionSpace(mesh, BrokenElement(FiniteElement(
            "FB", cell=mesh.ufl_cell(), degree=facet_degree + dim, variant="integral")))
    metadata = {"quadrature_degree": 2 * (max(space_degree, facet_degree) + dim + 1)}
    dx_q, ds_q, dS_q = dx(metadata=metadata), ds(metadata=metadata), dS(metadata=metadata)
    quadrature = list(zip(*_gauss(npoints)))

    # sum_{F in dK} (R_{K,F}, w)_F, with a separate trace R_{K,F} on each side of an interior facet
    def on_facets(density, test):
        return density * test * ds_q + (density("+") * test("+") + density("-") * test("-")) * dS_q

    def in_time(functions, x):
        return _combine(_lagrange(time_nodes, x)[0], functions)

    # (1 - tau) on the incoming face t_{n-1}, tau on the outgoing face t_n
    def face_factor(x):
        return x if endpoint == 1 else 1.0 - x

    # volume: (R_vol, B_{K,n} Phi) = rho(B_{K,n} Phi),  B_{K,n} = b_K 4 tau (1 - tau)
    W = MixedFunctionSpace([density_space] * len(time_nodes))
    trial, tests = TrialFunctions(W), TestFunctions(W)
    a = L = 0
    for x, w in quadrature:
        t, c = Constant(t0 + k * x), Constant(k * w)
        R = sum(Constant(float(l)) * r for l, r in zip(_lagrange(time_nodes, x)[0], trial))
        for l, test in zip(_lagrange(time_nodes, x)[0], tests):
            v = Constant(4 * x * (1 - x) * float(l)) * bubble * test
            a += c * R * v * dx_q
            L += volume_action(x, c * v, t)
    volume = Function(W)
    solve(a == L, volume, solver_parameters=RECOVERY_SOLVER)
    volume = list(volume.subfunctions)

    # spatial facets: (R_sf, C_sf Psi) = rho(C_sf Psi) - (R_vol, C_sf Psi)
    W = MixedFunctionSpace([facet_space] * len(time_nodes))
    trial, tests = TrialFunctions(W), TestFunctions(W)
    a = L = 0
    for x, w in quadrature:
        t, c = Constant(t0 + k * x), Constant(k * w)
        R = sum(Constant(float(l)) * r for l, r in zip(_lagrange(time_nodes, x)[0], trial)) / cone
        for l, test in zip(_lagrange(time_nodes, x)[0], tests):
            v = Constant(4 * x * (1 - x) * float(l)) * test
            a += c * on_facets(R, v)
            L += volume_action(x, c * v, t) - c * in_time(volume, x) * v * dx_q
    facet = Function(W)
    solve(a == L, facet, solver_parameters=RECOVERY_SOLVER)
    facet = [f / cone for f in facet.subfunctions]

    # temporal face: (R_tf, b_K chi) = rho(E_tf chi) - (R_vol, E_tf chi)
    trial, test = TrialFunction(density_space), TestFunction(density_space)
    a = trial * bubble * test * dx_q
    L = temporal_action(bubble * test)
    for x, w in quadrature:
        t, c = Constant(t0 + k * x), Constant(k * w)
        v = Constant(face_factor(x)) * bubble * test
        L += volume_action(x, c * v, t) - c * in_time(volume, x) * v * dx_q
    temporal = Function(density_space)
    solve(a == L, temporal, solver_parameters=RECOVERY_SOLVER)

    # mixed ridge: remove the vol, sf and tf parts from rho(E_xt psi)
    trial, test = TrialFunction(facet_space), TestFunction(facet_space)
    a = on_facets(trial / cone, test)
    L = temporal_action(test) - temporal * test * dx_q
    for x, w in quadrature:
        t, c = Constant(t0 + k * x), Constant(k * w)
        v = Constant(face_factor(x)) * test
        L += volume_action(x, c * v, t) - c * in_time(volume, x) * v * dx_q
        L -= c * on_facets(in_time(facet, x), v)
    ridge = Function(facet_space)
    solve(a == L, ridge, solver_parameters=RECOVERY_SOLVER)

    def pair(weight, face):
        """sum over the entities of Q_{K,n} of (density, weight), per cell K."""
        cell = TestFunction(FunctionSpace(mesh, "DG", 0))

        def facets(density, w_):       # an interior facet F is shared by K and K'
            return density * w_ * cell * ds_q + avg(density * w_) * (cell("+") + cell("-")) * dS_q

        form = temporal * weight(face) * cell * dx_q + facets(ridge / cone, weight(face))
        for x, w in quadrature:
            c = Constant(k * w)
            form += c * in_time(volume, x) * weight(x) * cell * dx_q + facets(in_time(facet, x), c * weight(x))
        return numpy.asarray(assemble(form).dat.data_ro, dtype=float).copy()

    return pair


def bubble_cone_indicators(P, nodes, U, U_in, e_z, npoints, degrees=None, Z=None, e_u=None, z_T=None):
    """eta_{K,n}: recovered primal densities paired with e_z;
    with Z, e_u, z_T also the recovered adjoint densities paired with e_u and both halved."""
    V, m = P["V"], P["m"]
    u, v, t_user = P["u"], P["v"], P["t"]
    if degrees is None:
        p = V.ufl_element().degree()
        degrees = (p, p, P["q"] + 1)
    N = len(U)
    eta = []
    for n in range(N):
        t0, k = nodes[n], nodes[n + 1] - nodes[n]

        def weight(pairing):
            functions, coefficients = pairing
            return lambda x: _combine(coefficients(x)[:, 0], functions)

        # primal residual rho(u_h), temporal term on the incoming face t_{n-1}^+
        def primal_volume(x, test, t, n=n, k=k):
            return -replace(m, {u: _at(U[n], x, k), v: test}) \
                - replace(P["F_x"], {u: _at(U[n], x), v: test, t_user: t})

        def primal_face(test, n=n):
            return -replace(m, {u: _at(U[n], 0.0) - U_in[n], v: test})

        pair = recover_densities(V.mesh(), t0, k, primal_volume, primal_face, 0, npoints, degrees)
        values = pair(weight(e_z[n]), 0.0)

        if Z is not None:
            # adjoint residual rho*(u_h, z_h), temporal term on the outgoing face t_n^-
            z_next = z_T if n == N - 1 else _at(Z[n + 1], 0.0, reverse=True)

            def adjoint_volume(x, test, t, n=n, k=k):
                form = replace(m, {u: test, v: _at(Z[n], x, k, reverse=True)}) \
                    - replace(P["A"], {v: test, P["z"]: _at(Z[n], x, reverse=True),
                                       u: _at(U[n], x), t_user: t})
                if P["dj"] is not None:
                    form += replace(P["dj"], {v: test, u: _at(U[n], x), t_user: t})
                return form

            def adjoint_face(test, n=n, z_next=z_next):
                return replace(m, {u: test, v: z_next - _at(Z[n], 1.0, reverse=True)})

            pair = recover_densities(V.mesh(), t0, k, adjoint_volume, adjoint_face, 1, npoints, degrees)
            values = 0.5 * values + 0.5 * pair(weight(e_u[n]), 1.0)
        eta.append(values)
    return numpy.array(eta)


def mark(eta_Kn, theta):
    """Slabs to bisect, as mark_split of the space-time solver with the spatial part off:
    slab n carries s_n = sum_K |eta_{K,n}|; slabs are ranked by max_K |eta_{K,n}| and
    taken until they carry theta * sum_n s_n."""
    values = numpy.abs(eta_Kn).sum(axis=1)
    order = numpy.argsort(-numpy.abs(eta_Kn).max(axis=1))
    count = int(numpy.searchsorted(numpy.cumsum(values[order]), theta * values.sum()) + 1)
    return set(order[:count].tolist())


def refine_time(nodes, marked):
    """Bisect the marked slabs."""
    new = [nodes[0]]
    for n in range(len(nodes) - 1):
        if n in marked:
            new.append(0.5 * (nodes[n] + nodes[n + 1]))
        new.append(nodes[n + 1])
    return numpy.array(new)


# ----------------------------------------------------------------------------------------
# Reference value and output
# ----------------------------------------------------------------------------------------
def reference_goal(problem, mesh, T, t, dt, nsteps=256, order=2, npoints=12):
    """J_ref - J_h measures the time error only."""
    F, u, bcs, goal = problem(mesh)[:4]
    t.assign(0.0)
    dt.assign(T / nsteps)
    stepper = TimeStepper(F, DiscontinuousGalerkinScheme(order, quadrature_degree=2 * order + 6),
                          t, dt, u, bcs=bcs, solver_parameters=LU)
    s, w = _gauss(npoints)
    phi = numpy.asarray(stepper.el.tabulate(0, [(x,) for x in s])[(0,)])
    Us, ts = Function(u.function_space()), Constant(0.0)
    j = replace(goal, {u: Us, t: ts})
    value = 0.0
    for n in range(nsteps):
        t0 = float(t)
        stepper.advance()
        for jj in range(len(s)):
            Us.assign(sum(float(a) * x for a, x in zip(phi[:, jj], stepper.stages.subfunctions)))
            ts.assign(t0 + float(dt) * s[jj])
            value += float(dt) * w[jj] * assemble(j)
        t.assign(t0 + float(dt))
    return value


def print_estimates(estimates):
    keys = [("iteration", "{:>3d}"), ("Nt", "{:>4d}"), ("kmin", "{:>9.2e}"), ("J_h", "{:>14.8e}"),
            ("error", "{:>10.3e}"), ("eta", "{:>10.3e}"), ("Ieff", "{:>6.3f}"),
            ("gap", "{:>8.1e}"), ("eta_rho", "{:>10.3e}"), ("eta_rho_star", "{:>12.3e}"),
            ("time_adjoint", "{:>7.2f}"), ("marked", "{:>4d}")]
    keys = [(k, f) for k, f in keys if any(k in row for row in estimates)]
    print("  ".join(f"{k:>{len(f.format(0) if 'd' in f else f.format(0.0))}}" for k, f in keys))
    for row in estimates:
        print("  ".join(f.format(row[k]) for k, f in keys))
