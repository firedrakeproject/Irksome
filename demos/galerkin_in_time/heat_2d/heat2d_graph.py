from firedrake import FunctionPlotter, tripcolor
from heat2d import solve_heat_2d_forced
from irksome.fetsome.fetutils import translate_generator
from argparse import ArgumentParser

# Parser setup to run the script
parser = ArgumentParser("python3 heat2d_graph.py", description="Solve and plot a movie of the 2d forced heat equation "
                        "with continuous time elements (cPG)")
parser.add_argument("plot_file", type=str, nargs=1,
                    help="The name of the file that will hold the plot (including extension)")
parser.add_argument("spatial_elements", type=int, nargs=1,
                    help="Number of spatial elements per spatial direction to solve the problem (sugg. 200)")
parser.add_argument("dt", type=float, nargs=1,
                    help="Timestep size for the temporal discretisation (sugg. 0.25)")
parser.add_argument("t_max", type=float, nargs=1,
                    help="Total time of solution (sugg 5.0)")
parser.add_argument("kt", type=int, nargs=1,
                    help="Polynomial degree of time finite element (sugg. 1)")

if __name__ == "__main__":
    args = parser.parse_args()
    plot_file = args.plot_file[0]
    Ns = args.spatial_elements[0]
    dt = args.dt[0]
    tmax = args.t_max[0]
    kt = args.kt[0]

    us = solve_heat_2d_forced(Ns, dt, tmax, kt, "CPG", info=True)

    print("Making the movie")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    steps = int(tmax / dt)
    nsp = 1024
    plotter = FunctionPlotter(us[0].function_space().mesh(), num_sample_points=nsp)

    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    colors = tripcolor(us[0], num_sample_points=nsp, vmin=0., vmax=4., axes=axes)
    fig.colorbar(colors)

    def animate(u):
        colors.set_array(plotter(u))

    interval = 1e1 * (steps + 1) * dt
    animation = FuncAnimation(fig, animate, frames=us, interval=interval)
    animation.save(plot_file, writer="ffmpeg")

