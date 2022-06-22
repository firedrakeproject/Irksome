from firedrake import *
from transport2d import solve_transport_2d
from irksome.fetsome.timenorm import time_errornorm
from irksome.fetsome.timequadrature import make_gauss_time_quadrature
from argparse import ArgumentParser

# Parser setup to run the script
parser = ArgumentParser("python3 transport2d_error.py", description="Compute the error for the analytic "
                        "example for the 1D heat equation.")
parser.add_argument("spatial_elements", type=int, nargs=1,
                    help="Number of spatial elements per spatial direction to solve the problem (sugg. 150)")
parser.add_argument("dt", type=float, nargs=1,
                    help="Timestep size for the temporal discretisation (sugg. 0.5)")
parser.add_argument("t_max", type=float, nargs=1,
                    help="Total time of solution (sugg 4.0)")
parser.add_argument("kt", type=int, nargs=1,
                    help="Polynomial degree of time finite element (sugg. 1)")

if __name__ == "__main__":
    # Parse the arguments for the script (including number of spatial elements, timestep,
    # total time, time basis degree)
    args = parser.parse_args()
    Ns = args.spatial_elements[0]
    dt = args.dt[0]
    tmax = args.t_max[0]
    kt = args.kt[0]

    us = solve_transport_2d(Ns, dt, tmax, kt, "CPG", info=True)

    time_quadrature = make_gauss_time_quadrature(7)
    V = us[0].function_space()
    x, y = SpatialCoordinate(V.mesh())

    t = Constant(0.)
    u_exact_expr = 5*(sin(pi/10*(x - 2.5*t))**2)*sin(pi/5*y)

    # Compute the error norm of the solution on kt-degree time elements
    T = FunctionSpace(UnitIntervalMesh(1), "CG", kt)
    error = time_errornorm(u_exact_expr, us, time_quadrature, dt, t, T)

    print("Error:", error)