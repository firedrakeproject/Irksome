from firedrake import *
from irksome.fetsome.fetutils import translate_generator
from burgers_nu import solve_burgers_nu
from argparse import ArgumentParser

import numpy as np

# Parser setup to run the script
parser = ArgumentParser("python3 burgers_nu_graph.py", description="Solve and plot a viscous 1d Burgers equation "
                        "on continuous spatial elements, continuous time elements (cPG)")
parser.add_argument("plot_file", type=str, nargs=1,
                    help="The name of the file that will hold the plot (including extension)")
parser.add_argument("spatial_elements", type=int, nargs=1,
                    help="Number of spatial elements per spatial direction to solve the problem (sugg. 200)")
parser.add_argument("dt", type=float, nargs=1,
                    help="Timestep size for the temporal discretisation (sugg. 0.0333)")
parser.add_argument("t_max", type=float, nargs=1,
                    help="Total time of solution (sugg 0.5)")
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
    us = solve_burgers_nu(Ns, dt, tmax, kt, "CPG", info=True)

    # Prepare the analytical solution to graph alongside it
    V = us[0].function_space()
    x = SpatialCoordinate(V.mesh())[0]

    t = Constant(0.)

    snapnum = len(us)
    # Do the graphing functionality
    plotsnum = int(sqrt(snapnum))

    horiz_ticks = np.linspace(0., 2., 5)
    vert_ticks = np.linspace(-1.5, 1.5, 7)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(plotsnum, plotsnum, sharex=True, sharey=True)
    fig.set_size_inches(12, 7)
    for i in range(plotsnum):
        for j in range(plotsnum):
            axes[i,j].set_ylim([-1.5, 1.5])
            axes[i,j].set_xticks(horiz_ticks, minor=True)
            axes[i,j].set_yticks(vert_ticks, minor = True)
            plot(us[plotsnum*i + j], axes=axes[i, j], color="blue", linewidth=0.8)
            axes[i,j].legend(["$u_{kh}$"])
            axes[i,j].set_title("$t = " + str((plotsnum*i + j) * dt/kt) + "$")
            axes[i,j].grid(linewidth=0.4)

    fig.supxlabel("Spatial Coordinate $x$")
    fig.supylabel("$u(x, t)$")
    fig.tight_layout()

    plt.savefig(plot_file, dpi=300.)