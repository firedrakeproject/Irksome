from firedrake import FunctionPlotter, tripcolor
from fetsome.galerkin_in_time.transport_2d.transport2d import solve_transport_2d
from fetsome.fet.fetutils import translate_generator
from argparse import ArgumentParser

# Parser setup to run the script
parser = ArgumentParser("python3 transport2d_graph.py", description="Solve and plot a movie of the 2D transport equation "
                        "on a periodic boundary")
parser.add_argument("plot_file", type=str, nargs=1,
                    help="The name of the file that will hold the plot (including extension)")
parser.add_argument("spatial_elements", type=int, nargs=1,
                    help="Number of spatial elements per spatial direction to solve the problem (sugg. 200)")
parser.add_argument("dt", type=float, nargs=1,
                    help="Timestep size for the temporal discretisation (sugg. 0.5)")
parser.add_argument("t_max", type=float, nargs=1,
                    help="Total time of solution (sugg 4.0)")
parser.add_argument("kt", type=int, nargs=1,
                    help="Polynomial degree of time finite element (sugg. 1)")
parser.add_argument("generator", type=str, nargs=1, choices=("petrov", "tdg"),
                    help="Type of time form generator to pass to the solver")

if __name__ == "__main__":
    args = parser.parse_args()
    plot_file = args.plot_file[0]
    Ns = args.spatial_elements[0]
    dt = args.dt[0]
    tmax = args.t_max[0]
    kt = args.kt[0]
    generator = args.generator[0]

    generator_code = translate_generator[generator]
    us = solve_transport_2d(Ns, dt, tmax, kt, generator_code, info=True)

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

