from firedrake import *
from irksome import GaussLegendre, Dt
from irksome.stage_derivative import getForm
import pytest


@pytest.mark.parametrize("num_stages", [1, 2])
def test_multidomain_getForm(num_stages):
    t = Constant(0)
    dt = Constant(0.1)
    mesh1 = UnitSquareMesh(1, 1)
    mesh2 = Submesh(mesh1, mesh1.topological_dimension-1, 1, label_name="exterior_facets")

    V1 = FunctionSpace(mesh1, "CG", 1)
    V2 = FunctionSpace(mesh2, "CG", 1)
    Z = V1 * V2

    z = Function(Z)
    u1, u2 = split(z)
    v1, v2 = TrialFunctions(Z)

    dx1 = dx(mesh1)
    dx2 = dx(mesh2)
    dx12 = Measure("ds", domain=mesh1, intersect_measures=[dx2])

    F = (inner(Dt(u1), v1) * dx1
         + inner(Dt(u2), v2) * dx2
         + inner(u1, v1) * dx1
         + inner(u2, v2) * dx2
         + inner(u1-u2, v1-v2) * dx12)

    butch = GaussLegendre(num_stages)
    Zbig = MixedFunctionSpace([Z]*num_stages)
    stages = Function(Zbig)

    Fnew, bcs = getForm(F, butch, t, dt, z, stages)
    assemble(Fnew)
