from firedrake import *
from fetsome.galerkin_in_time.heat_tdg.heat_tdg import solve_heat_tdg
import numpy as np
from argparse import ArgumentParser
from fetsome.fet.fetutils import translate_generator

# Parser setup to run the script
parser = ArgumentParser("python3 heat_tdg_graph.py", description="Solve and plot the analytic "
                        "example for the 1D heat equation.")
parser.add_argument("plot_file", type=str, nargs=1,
                    help="The name of the file that will hold the plot (including extension)")
parser.add_argument("spatial_elements", type=int, nargs=1,
                    help="Number of spatial elements per spatial direction to solve the problem (sugg. 200)")
parser.add_argument("dt", type=float, nargs=1,
                    help="Timestep size for the temporal discretisation (sugg. 0.125)")
parser.add_argument("t_max", type=float, nargs=1,
                    help="Total time of solution (sugg 5.0)")
parser.add_argument("kt", type=int, nargs=1,
                    help="Polynomial degree of time finite element (sugg. 1)")

if __name__ == "__main__":
    # Parse the arguments for the script (including number of spatial elements, timestep,
    # total time, time basis degree)
    args = parser.parse_args()
    plot_file = args.plot_file[0]
    Ns = args.spatial_elements[0]
    dt = args.dt[0]
    tmax = args.t_max[0]
    kt = args.kt[0]

    # Get the approximated solution
    print("Solving the equation...")
    us = solve_heat_tdg(Ns, dt, tmax, kt, "DG", info=True)

    # Prepare the analytical solution to graph alongside it
    V = us[0].function_space()
    x = SpatialCoordinate(V.mesh())[0]

    t = Constant(0.)
    u_exact_expr = (1./2.)*cos((2*pi/7.)*x)*exp((-4.*pi**2/7.**2)*t)

    snapnum = len(us)
    u_exact = interpolate(u_exact_expr, V)
    us_exact = []
    for i in range(snapnum):
        t.assign(dt/kt * i)
        us_exact.append(interpolate(u_exact_expr, V))

    # Do the graphing functionality
    plotsnum = int(sqrt(snapnum))

    horiz_ticks = np.linspace(0., 7., 7)
    vert_ticks = np.linspace(-0.5, 0.5, 5)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(plotsnum, plotsnum, sharex=True, sharey=True)
    fig.set_size_inches(10, 6)
    for i in range(plotsnum):
        for j in range(plotsnum):
            axes[i,j].set_ylim([-0.5, 0.5])
            axes[i,j].set_xticks(horiz_ticks, minor=True)
            axes[i,j].set_yticks(vert_ticks, minor = True)
            plot(us_exact[plotsnum*i + j], axes=axes[i, j], color="red", linewidth=0.8, linestyle="dotted")
            plot(us[plotsnum*i + j], axes=axes[i, j], color="black", linewidth=0.8)
            axes[i,j].legend(["$u$ Exact", "$u_{kh}$"])
            axes[i,j].set_title("$t = " + str((plotsnum*i + j) * dt/kt) + "$")
            axes[i,j].grid(linewidth=0.4)

    fig.supxlabel("Spatial Coordinate $x$")
    fig.supylabel("$u(x, t)$")
    fig.tight_layout()

    plt.savefig(plot_file, dpi=300.)