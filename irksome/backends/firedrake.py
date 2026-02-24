"""Firedrake backend for Irksome"""

import firedrake
import ufl
from ..tools import get_stage_space

def function_space(u: ufl.Coefficient) -> firedrake.FunctionSpace:
    return u.function_space()


def get_stages(V: firedrake.FunctionSpace, num_stages: int) -> firedrake.Function:
    """
    Given a function space for a single time-step, get a duplicate of this space,
    repeated `num_stages` times.

    Args:
        V: Space for single step
        num_stages: Number of stages

    Returns:
        A coefficient in the new function space
    """
    Vbig = get_stage_space(V, num_stages)
    return firedrake.Function(Vbig)



class MeshConstant(object):
    def __init__(self, msh: ufl.Mesh):
        self.msh = ufl.domain.as_domain(msh)
        self.V = firedrake.FunctionSpace(self.msh, 'R', 0)

    def Constant(self, val=0.0)->ufl.Coefficient:
        return firedrake.Function(self.V).assign(val)

def get_mesh_constant(MC: MeshConstant|None):
    return MC.Constant if MC else firedrake.Constant
