#from firedrake import *
#from firedrake.adjoint import *
#from irksome import Dt, RadauIIA, TimeStepper
#
# def test_adjoint_diffusivity():
#     print("Script starting...")
#     msh = UnitIntervalMesh(4)
#     V = FunctionSpace(msh, "CG", 1)
# 
#     R = FunctionSpace(msh, "R", 0)
#     kappa = Function(R).assign(2.0)
#     c = Control(kappa)
# 
#     v = TestFunction(V)
#     u = Function(V)
#     u.assign(4)
# 
#     stages = Function(V) 
# 
#     # Succeed - Yes!
#     continue_annotation()
#     _ = stages.subfunctions
# 
#     # Fail - Yes!
#     # _ = stages.subfunctions
#     # continue_annotation()
# 
#     with set_working_tape() as tape:
#         stages.assign(kappa)
#         u += stages.subfunctions[0]
#         J = assemble(inner(u, u) * dx)
#         rf = ReducedFunctional(J, c, tape=tape)
#     pause_annotation()
# 
#     rf.derivative()
# 
#     nu = Function(R).assign(3)
#     print(f"{rf(kappa) = }")
#     print(f"{rf(nu) = }")
# 
#     assert abs(rf(kappa) - rf(nu)) > 1e-8

