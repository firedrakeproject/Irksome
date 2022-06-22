from firedrake import *
from irksome.fetsome.fetutils import translate_generator
from burgers_exact import solve_burgers_exact
from argparse import ArgumentParser

import numpy as np

# Parser setup to run the script
parser = ArgumentParser("python3 burgers_exact_graph.py", description="Solve and plot an exact viscous 1d Burgers equation")
parser.add_argument("plot_file", type=str, nargs=1,
                    help="The name of the file that will hold the plot (including extension)")
parser.add_argument("spatial_elements", type=int, nargs=1,
                    help="Number of spatial elements per spatial direction to solve the problem (sugg. 100)")
parser.add_argument("dt", type=float, nargs=1,
                    help="Timestep size for the temporal discretisation (sugg. 0.1)")
parser.add_argument("t_max", type=float, nargs=1,
                    help="Total time of solution (sugg 0.8)")
parser.add_argument("kt", type=int, nargs=1,
                    help="Polynomial degree of time finite element (sugg. 1)")
parser.add_argument("generator", type=str, nargs=1, choices=("standard","nostep","petrov"),
                    help="Type of time form generator to pass to the solver")

if __name__ == "__main__":
    # Parse the arguments for the script (including number of spatial elements, timestep,
    # total time, time basis degree)
    args = parser.parse_args()
    plot_file = args.plot_file[0]
    Ns = args.spatial_elements[0]
    dt = args.dt[0]
    tmax = args.t_max[0]
    kt = args.kt[0]
    generator = args.generator[0]

    # Parse type of form generator
    generator_code = translate_generator[generator]

    # Get the approximated solution
    print("Solving the equation...")
    us = solve_burgers_exact(Ns, dt, tmax, kt, generator_code, info=True)

    # Prepare the analytical solution to graph alongside it
    V = us[0].function_space()
    x = SpatialCoordinate(V.mesh())[0]

    t = Constant(0.)
    nu = 0.05
    alpha = 1.5
    beta = 1.55
    k = pi / 2.0
    u_exact_expr = (2*nu*alpha*k*exp(-nu*k**2*t)*sin(k*x)) / (beta + alpha*exp(-nu*k**2*t)*cos(k*x))

    snapnum = len(us)
    u_exact = interpolate(u_exact_expr, V)
    us_exact = []
    for i in range(snapnum):
        t.assign(dt/kt * i)
        us_exact.append(interpolate(u_exact_expr, V))

    snapnum = len(us)
    # Do the graphing functionality
    plotsnum = int(sqrt(snapnum))

    horiz_ticks = np.linspace(0, 8, 9)
    vert_ticks = np.linspace(-0.75, 0.75, 7)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(plotsnum, plotsnum, sharex=True, sharey=True)
    fig.set_size_inches(12, 7)
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