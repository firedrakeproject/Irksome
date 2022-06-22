from firedrake import *
from irksome.fetsome.fetutils import translate_generator
from irksome.fetsome.timenorm import time_errornorm
from irksome.fetsome.timequadrature import time_gauss_quadrature_overkill
from heat2d import solve_heat_2d_forced
from argparse import ArgumentParser

# Parser setup to run the script
parser = ArgumentParser("python3 heat2d_error.py", description="Compute the error for the the 2d forced heat equation")
parser.add_argument("spatial_elements", type=int, nargs=1,
                    help="Number of spatial elements per spatial direction to solve the problem (sugg. 200)")
parser.add_argument("dt", type=float, nargs=1,
                    help="Timestep size for the temporal discretisation (sugg. 0.25)")
parser.add_argument("t_max", type=float, nargs=1,
                    help="Total time of solution (sugg 5.0)")
parser.add_argument("kt", type=int, nargs=1,
                    help="Polynomial degree of time finite element (sugg. 1)")
parser.add_argument("generator", type=str, nargs=1, choices=("standard","nostep","petrov"),
                    help="Type of time form generator to pass to the solver")

if __name__ == "__main__":
    # Parse the arguments for the script (including number of spatial elements, timestep,
    # total time, time basis degree)
    args = parser.parse_args()
    Ns = args.spatial_elements[0]
    dt = args.dt[0]
    tmax = args.t_max[0]
    kt = args.kt[0]
    generator = args.generator[0]

    # Parse type of form generator
    generator_code = translate_generator[generator]

    us = solve_heat_2d_forced(Ns, dt, tmax, kt, generator_code, info=True)

    # Prepare the snapshots for the analytic solution:
    #       u(x,y,t) = pi * sin^2(pi*x) * sin^2(pi*y) * e^(-1/2 t)
    time_quadrature = time_gauss_quadrature_overkill()
    V = us[0].function_space()
    x, y = SpatialCoordinate(V.mesh())

    t = Constant(0.)
    u_exact_expr = pi*(sin(pi*x)**2)*(sin(pi*y)**2) * exp(-1/2*t)

    # Compute the error norm of the solution on kt-degree time elements
    T = FunctionSpace(UnitIntervalMesh(1), "CG", kt)
    error = time_errornorm(u_exact_expr, us, time_quadrature, dt, t, T)

    print("Error:", error)