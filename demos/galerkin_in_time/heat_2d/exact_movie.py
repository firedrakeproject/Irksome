from firedrake import *
from argparse import ArgumentParser

# Parser setup to run the script
parser = ArgumentParser("python3 exact_movie.py", description="Solve and plot a movie of the exact solution "
                        "to the 2d forced heat equation")
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

    mesh = SquareMesh(Ns, Ns, 2)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    t = Constant(0.)
    u_exact_expr = pi*(sin(pi*x)**2)*(sin(pi*y)**2) * exp(-1/2*t)

    steps = int(tmax / dt)
    us = []
    for i in range(0, (steps + 1)*kt):
        ti = i/kt * dt
        us.append(interpolate(replace(u_exact_expr, {t: Constant(ti)}), V))


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