from firedrake import *
from .heat1d import solve_heat_analytic
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from irksome.fetsome.fetutils import translate_generator

# Quick helper function to integrate a function in the spatial dimension
def energy(u):
    return assemble(u * dx)

parser = ArgumentParser("python3 heat1d_conserved.jpg", description="Draw a graph of the total energy over time "
                        "for the numerical example of the 1d analytic heat equation.")
parser.add_argument("plot_file", type=str, nargs=1,
                    help="Name of the where the energy plot will be saved.")
parser.add_argument("spatial_elements", type=int, nargs=1,
                    help="Number of spatial elements per spatial direction to solve the problem (sugg. 200)")
parser.add_argument("dt", type=float, nargs=1,
                    help="Timestep size for the temporal discretisation (sugg. 2)")
parser.add_argument("t_max", type=float, nargs=1,
                    help="Total time of solution (sugg 20.0)")
parser.add_argument("kt", type=int, nargs=1,
                    help="Polynomial degree of time finite element (sugg. 1)")
parser.add_argument("generator", type=str, nargs=1, choices=("standard","nostep","petrov"),
                    help="Type of time form generator to pass to the solver")


if __name__ == "__main__":
    # Parse the arguments (including number of spatial elements, timestep size,
    # total time, time polynomial degree)
    args = parser.parse_args()
    plot_file = args.plot_file[0]
    Ns = args.spatial_elements[0]
    dt = args.dt[0]
    tmax = args.t_max[0]
    kt = args.kt[0]
    generator = args.generator[0]

    # Extract the generator code
    generator_code = translate_generator[generator]

    us = solve_heat_analytic(Ns, dt, tmax, kt, generator_code, info=True)

    # Calculate the energies (total integral over domain) for each of the interesting timesteps
    ts = [dt * i for i in range(len(us))]
    energies = [energy(u) for u in us]


    # Plot the graph
    fig, axes = plt.subplots()
    fig.set_size_inches(12, 7)
    axes.set_ylim([-0.1, 0.1])
    axes.set_title("Total Energy over Time")
    axes.set_xlabel("t")
    axes.set_ylabel("Energy")
    axes.grid()
    axes.scatter(ts, energies, c="r")
    axes.legend(["Energy"])

    plt.savefig(plot_file, dpi=300)
