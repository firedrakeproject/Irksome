from firedrake import *
from fetsome.galerkin_in_time.heat_mms.forced import solve_heat_forced
from fetsome.fet.timequadrature import time_gauss_quadrature_overkill
from fetsome.fet.timenorm import time_errornorm
from fetsome.fet.fetutils import translate_generator
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser("python3 forced_error_graph.py", description="Draw a graph of the error for the MMS "
                        "example of the 1d heat equation.")
parser.add_argument("plot_file", type=str, nargs=1,
                    help="Name of the file that will contain the error plot.")
parser.add_argument("spatial_elements", type=int, nargs=1,
                    help="Number of spatial elements per spatial direction to solve the problem (sugg. 200)")
parser.add_argument("t_max", type=float, nargs=1,
                    help="Total time of solution (sugg 4.0)")
parser.add_argument("kt", type=int, nargs=1,
                    help="Polynomial degree of time finite element (sugg. 1)")
parser.add_argument("generator", type=str, nargs=1, choices=("petrov", "tdg"),
                    help="Type of time form generator to pass to the solver")

if __name__ == "__main__":
    # Parse the script arguments (including number of spatial elements, total time, time basis degree)
    args = parser.parse_args()
    plot_file = args.plot_file[0]
    Ns = args.spatial_elements[0]
    tmax = args.t_max[0]
    kt = args.kt[0]
    generator = args.generator[0]

    # Parse type of form generator
    generator_code = translate_generator[generator]

    dts = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1., 2.]
    # Safety check on minimum timestep
    if tmax < max(dts):
        raise AssertionError("Timestep size", max(dts), "larger than total time", tmax)

    time_quadrature = time_gauss_quadrature_overkill()

    es = []
    for dt in dts:
        print("Running error estimate for dt =", dt)
        us = solve_heat_forced(Ns, dt, tmax, kt, generator_code, info=True)

        V = us[0].function_space()
        x = SpatialCoordinate(V.mesh())[0]

        t = Constant(0.)
        u_exact_expr = (1/100 * x**3 - 3/20 * x**2 + 5) * exp(-t)


        # Compute the error norm of the solution
        T = FunctionSpace(UnitIntervalMesh(1), "CG", kt)
        error = time_errornorm(u_exact_expr, us, time_quadrature, dt, t, T)

        print("Finished computing error for dt =", dt)
        print("Error:", error, "[ dt =", dt,"]")
        es.append(error)
    
    # Calculate the convergence rates for each timestep jump for all dts
    from math import log
    print("Errors:", es)
    qs = [log(es[i+1]/es[i]) / log(dts[i+1]/dts[i]) for i in range(len(es) - 1)]
    print("Convergence rates:", qs)
    rate_str = ", ".join(["%.2f" % q for q in qs])

    # Expected rate of convergence for problem and reference line
    expected_conv = kt + 1
    shift = max(es)
    ref_pts = [shift * tau**expected_conv for tau in dts]
    
    # Draw the loglog plot
    fig, axes = plt.subplots()
    fig.set_size_inches(4, 3)
    axes.set_ylim([10e-11, 10e2])
    axes.invert_xaxis()
    axes.loglog(dts, es, label="$L^2$ Error", color="purple")
    axes.loglog(dts, ref_pts, label="$L^2$ Reference $O(dt^" + str(expected_conv) + ")$",
                linestyle="dotted")

    axes.set_xlabel("Timestep Size $dt$")
    axes.set_ylabel("Spacetime $L^2$ Error")
    axes.grid(linewidth=0.2)
    axes.legend()

    fig.tight_layout()

    plt.savefig(plot_file, dpi=300.)