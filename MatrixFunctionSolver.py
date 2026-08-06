import sys, petsc4py
petsc4py.init(sys.argv)
import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
from firedrake import (FunctionSpace, exp, pi, div, cos, norm, SpatialCoordinate,
                       Function, Cofunction, TrialFunction, errornorm, Constant,
                       UnitIntervalMesh, inner, TestFunction, dx, assemble, grad)
from irksome import Dt, expand_time_derivatives
import scipy as sp
import numpy as np
from ufl import replace
from firedrake.petsc import OptionsManager, flatten_parameters
from firedrake.exceptions import ConvergenceError


class MInvK:

    def __init__(self, a):
        self.a = a   # bilinear form
        trial, test = a.arguments()
        self.V = trial.function_space()
        self.f = Cofunction(self.V.dual())
        self.K = assemble(self.a).petscmat

    def mult(self, mat, X, Y):
        with self.f.dat.vec_wo as Z:
            self.K.mult(X, Z)  # gives Z = K * X, takes Z = K * X and loads it into f
        g = self.f.riesz_representation()  # gives g, M^-1Ku
        with g.dat.vec_ro as D:
            D.copy(Y)   # puts M^-1Ku into Y


def uexact(x, t):
    return exp(-t) * cos(pi * x)


class ExponentialRKButcherTableau:
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c
        self.num_stages = self.A.shape[0]

        # check lengths of A, b, & c
        len_A = len(self.A)
        len_b = len(self.b)
        len_c = len(self.c)
        if len_b == len_c:
            if len_A == len_b:
                return "lengths are compatible"
            else:
                return "A does not match up w/ b & c"
        else:
            return "b & c are not the same length"

        @property
        def num_stages(self):
            """Return the number of stages the method has."""
            return len(self.b)

        @property
        def is_explicit(self):
            return np.allclose(np.triu(self.A), 0)

        assert self.is_explicit()


class OneParameterFamily1(ExponentialRKButcherTableau):
    """Example 2.18 from Hochbruck and Ostermann"""
    def __init__(self, c2):
        c = np.array([0, c2])

        b1a = SLEPc.FN().create()
        b1a.setType('phi')
        b1a.setPhiIndex(1)
        b1b = SLEPc.FN().create()
        b1b.setType('phi')
        b1b.setPhiIndex(2)
        b1b.setScale(1, -1/Constant(c2))
        fn_b1 = SLEPc.FN().create()
        fn_b1.setType('combine')
        fn_b1.setCombineChildren(SLEPc.FN.CombineType.ADD, b1a, b1b)

        fn_b2 = SLEPc.FN().create()
        fn_b2.setType('phi')
        fn_b2.setPhiIndex(2)
        fn_b2.setScale(1, 1/Constant(c2))

        b = np.array([fn_b1, fn_b2], dtype="object")

        fn_a21 = SLEPc.FN().create()
        fn_a21.setType('phi')
        fn_a21.setPhiIndex(1)
        fn_a21.setScale(Constant(c2), Constant(c2))

        A = np.array([[0, 0], [fn_a21, 0]], dtype="object")

        super(OneParameterFamily1, self).__init__(A, b, c)


class CoxAndMatthews2002(ExponentialRKButcherTableau):
    """Example 2.19 from Hochbruck and Ostermann, 2.41"""
    def __init__(self, C):
        c = C  # numpy array

        b1a = SLEPc.FN().create()
        b1a.setType('phi')
        b1a.setPhiIndex(1)
        b1b = SLEPc.FN().create()
        b1b.setType('phi')
        b1b.setPhiIndex(2)
        b1b.setScale(1, -3)
        b1c = SLEPc.FN().create()
        b1c.setType('phi')
        b1c.setPhiIndex(3)
        b1c.setScale(1, 4)
        b1a_b1b = SLEPc.FN().create()
        b1a_b1b.setType('combine')
        b1a_b1b.setCombineChildren(SLEPc.FN.CombineType.ADD, b1a, b1b)
        fn_b1 = SLEPc.FN().create()
        fn_b1.setType('combine')
        fn_b1.setCombineChildren(SLEPc.FN.CombineType.ADD, b1a_b1b, b1c)

        b2a = SLEPc.FN().create()
        b2a.setType('phi')
        b2a.setPhiIndex(2)
        b2a.setScale(1, 2)
        b2b = SLEPc.FN().create()
        b2b.setType('phi')
        b2b.setPhiIndex(3)
        b2b.setScale(1, -4)
        fn_b2 = SLEPc.FN().create()
        fn_b2.setType('combine')
        fn_b2.setCombineChildren(SLEPc.FN.CombineType.ADD, b2a, b2b)

        fn_b3 = SLEPc.FN().create()
        fn_b3.setType('combine')
        fn_b3.setCombineChildren(SLEPc.FN.CombineType.ADD, b2a, b2b)

        b4b = SLEPc.FN().create()
        b4b.setType('phi')
        b4b.setPhiIndex(2)
        b4b.setScale(1, -1)
        fn_b4 = SLEPc.FN().create()
        fn_b4.setType('combine')
        fn_b4.setCombineChildren(SLEPc.FN.CombineType.ADD, b1c, b4b)
    
        b = np.array([fn_b1, fn_b2, fn_b3, fn_b4], dtype=object)

        fn_a21 = SLEPc.FN().create()
        fn_a21.setType('phi')
        fn_a21.setPhiIndex(1)
        fn_a21.setScale(Constant(c[1]), 0.5)

        fn_a32 = SLEPc.FN().create()
        fn_a32.setType('phi')
        fn_a32.setPhiIndex(1)
        fn_a32.setScale(Constant(c[2]), 0.5)

        fn_a43 = SLEPc.FN().create()
        fn_a43.setType('phi')
        fn_a43.setPhiIndex(1)
        fn_a43.setScale(Constant(c[2]), 1)

        ## previous combine mult videos
        # a41b = SLEPc.FN().create()
        # a41b.setType('phi')
        # a41b.setPhiIndex(0)
        # a41b.setScale(Constant(c[2]), 1)
        # a41a = SLEPc.FN().create()
        # a41a.setType('phi')
        # a41a.setPhiIndex(1)
        # a41a.setScale(Constant(c[2]), 0.5)
        # a41ab = SLEPc.FN().create()
        # a41ab.setType('combine')
        # a41ab.setCombineChildren(SLEPc.FN.CombineType.MULTIPLY, a41a, a41b)
        # a41d = SLEPc.FN().create()
        # a41d.setType('phi')
        # a41d.setPhiIndex(1)
        # a41d.setScale(Constant(c[2]), -0.5)
        # fn_a41 = SLEPc.FN().create()
        # fn_a41.setType('combine')
        # fn_a41.setCombineChildren(SLEPc.FN.CombineType.ADD, a41ab, a41d)

        a41a = SLEPc.FN().create()
        a41a.setType('phi')
        a41a.setPhiIndex(1)
        a41a.setScale(Constant(2*c[2]), 1)
        a41b = SLEPc.FN().create()
        a41b.setType('phi')
        a41b.setPhiIndex(1)
        a41b.setScale(Constant(c[2]), -1)
        fn_a41 = SLEPc.FN().create()
        fn_a41.setType('combine')
        fn_a41.setCombineChildren(SLEPc.FN.CombineType.ADD, a41a, a41b)

        A = np.array([[0, 0, 0, 0], [fn_a21, 0, 0, 0], [0, fn_a32, 0, 0], [fn_a41, 0, fn_a43, 0]], dtype=object)

        super(CoxAndMatthews2002, self).__init__(A, b, c)


class Krostgrad2005(ExponentialRKButcherTableau):
    """Example 2.19 from Hochbruck and Ostermann, 2.42"""
    def __init__(self, C):
        c = C

        fn_a21 = SLEPc.FN().create()
        fn_a21.setType('phi')
        fn_a21.setPhiIndex(1)
        fn_a21.setScale(Constant(c[1]), 0.5)

        a31a = SLEPc.FN().create()
        a31a.setType('phi')
        a31a.setPhiIndex(1)
        a31a.setScale(Constant(c[2]), 0.5)
        a31b = SLEPc.FN().create()
        a31b.setType('phi')
        a31b.setPhiIndex(2)
        a31b.setScale(Constant(c[2]), -1)
        fn_a31 = SLEPc.FN().create()
        fn_a31.setType('combine')
        fn_a31.setCombineChildren(SLEPc.FN.CombineType.ADD, a31a, a31b)

        fn_a32 = SLEPc.FN().create()
        fn_a32.setType('phi')
        fn_a32.setPhiIndex(2)
        fn_a32.setScale(Constant(c[2]), 1)

        a41a = SLEPc.FN().create()
        a41a.setType('phi')
        a41a.setPhiIndex(1)
        a41a.setScale(Constant(c[3]), 1)
        a41b = SLEPc.FN().create()
        a41b.setType('phi')
        a41b.setPhiIndex(2)
        a41b.setScale(Constant(c[3]), -2)
        fn_a41 = SLEPc.FN().create()
        fn_a41.setType('combine')
        fn_a41.setCombineChildren(SLEPc.FN.CombineType.ADD, a41a, a41b)

        fn_a43 = SLEPc.FN().create()
        fn_a43.setType('phi')
        fn_a43.setPhiIndex(2)
        fn_a43.setScale(Constant(c[3]), 2)

        A = np.array([[0, 0, 0, 0], [fn_a21, 0, 0, 0], [fn_a31, fn_a32, 0, 0], [fn_a41, 0, fn_a43, 0]], dtype=object)
        
        b1a = SLEPc.FN().create()
        b1a.setType('phi')
        b1a.setPhiIndex(1)
        b1b = SLEPc.FN().create()
        b1b.setType('phi')
        b1b.setPhiIndex(2)
        b1b.setScale(1, -3)
        b1c = SLEPc.FN().create()
        b1c.setType('phi')
        b1c.setPhiIndex(3)
        b1c.setScale(1, 4)
        b1a_b1b = SLEPc.FN().create()
        b1a_b1b.setType('combine')
        b1a_b1b.setCombineChildren(SLEPc.FN.CombineType.ADD, b1a, b1b)
        fn_b1 = SLEPc.FN().create()
        fn_b1.setType('combine')
        fn_b1.setCombineChildren(SLEPc.FN.CombineType.ADD, b1a_b1b, b1c)

        b2a = SLEPc.FN().create()
        b2a.setType('phi')
        b2a.setPhiIndex(2)
        b2a.setScale(1, 2)
        b2b = SLEPc.FN().create()
        b2b.setType('phi')
        b2b.setPhiIndex(3)
        b2b.setScale(1, -4)
        fn_b2 = SLEPc.FN().create()
        fn_b2.setType('combine')
        fn_b2.setCombineChildren(SLEPc.FN.CombineType.ADD, b2a, b2b)

        fn_b3 = SLEPc.FN().create()
        fn_b3.setType('combine')
        fn_b3.setCombineChildren(SLEPc.FN.CombineType.ADD, b2a, b2b)

        b4b = SLEPc.FN().create()
        b4b.setType('phi')
        b4b.setPhiIndex(2)
        b4b.setScale(1, -1)
        fn_b4 = SLEPc.FN().create()
        fn_b4.setType('combine')
        fn_b4.setCombineChildren(SLEPc.FN.CombineType.ADD, b1c, b4b)
    
        b = np.array([fn_b1, fn_b2, fn_b3, fn_b4], dtype=object)

        super(Krostgrad2005, self).__init__(A, b, c)


class HOFiveStage(ExponentialRKButcherTableau):
    """Example 2.19 from Hochbruck and Ostermann page 229"""
    def __init__(self, C):
        c = C

        fn_a21 = SLEPc.FN().create()
        fn_a21.setType('phi')
        fn_a21.setPhiIndex(1)
        fn_a21.setScale(Constant(c[1]), 0.5)

        a31a = SLEPc.FN().create()
        a31a.setType('phi')
        a31a.setPhiIndex(1)
        a31a.setScale(Constant(c[2]), 0.5)
        a31b = SLEPc.FN().create()
        a31b.setType('phi')
        a31b.setPhiIndex(2)
        a31b.setScale(Constant(c[2]), -1)
        fn_a31 = SLEPc.FN().create()
        fn_a31.setType('combine')
        fn_a31.setCombineChildren(SLEPc.FN.CombineType.ADD, a31a, a31b)

        fn_a32 = SLEPc.FN().create()
        fn_a32.setType('phi')
        fn_a32.setPhiIndex(2)
        fn_a32.setScale(Constant(c[2]), 1)

        a41a = SLEPc.FN().create()
        a41a.setType('phi')
        a41a.setPhiIndex(1)
        a41a.setScale(Constant(c[3]), 1)
        a41b = SLEPc.FN().create()
        a41b.setType('phi')
        a41b.setPhiIndex(2)
        a41b.setScale(Constant(c[3]), -2)
        fn_a41 = SLEPc.FN().create()
        fn_a41.setType('combine')
        fn_a41.setCombineChildren(SLEPc.FN.CombineType.ADD, a41a, a41b)

        fn_a42 = SLEPc.FN().create()
        fn_a42.setType('phi')
        fn_a42.setPhiIndex(2)
        fn_a42.setScale(Constant(c[3]), 1)

        fn_a43 = SLEPc.FN().create()
        fn_a43.setType('phi')
        fn_a43.setPhiIndex(2)
        fn_a43.setScale(Constant(c[3]), 1)

        a52a = SLEPc.FN().create()
        a52a.setType('phi')
        a52a.setPhiIndex(2)
        a52a.setScale(Constant(c[4]), .5)
        a52b = SLEPc.FN().create()
        a52b.setType('phi')
        a52b.setPhiIndex(3)
        a52b.setScale(Constant(c[3]), -1)
        a52c = SLEPc.FN().create()
        a52c.setType('phi')
        a52c.setPhiIndex(2)
        a52c.setScale(Constant(c[3]), .25)
        a52d = SLEPc.FN().create()
        a52d.setType('phi')
        a52d.setPhiIndex(3)
        a52d.setScale(Constant(c[4]), -.5)
        a52a_a52b = SLEPc.FN().create()
        a52a_a52b.setType('combine')
        a52a_a52b.setCombineChildren(SLEPc.FN.CombineType.ADD, a52a, a52b)
        a52a_a52b_a52c = SLEPc.FN().create()
        a52a_a52b_a52c.setType('combine')
        a52a_a52b_a52c.setCombineChildren(SLEPc.FN.CombineType.ADD, a52a_a52b, a52c)
        fn_a52 = SLEPc.FN().create()
        fn_a52.setType('combine')
        fn_a52.setCombineChildren(SLEPc.FN.CombineType.ADD, a52a_a52b_a52c, a52d)

        a52a2 = SLEPc.FN().create()
        a52a2.setType('phi')
        a52a2.setPhiIndex(2)
        a52a2.setScale(Constant(c[4]), -.5)
        a52b2 = SLEPc.FN().create()
        a52b2.setType('phi')
        a52b2.setPhiIndex(3)
        a52b2.setScale(Constant(c[3]), 1)
        a52c2 = SLEPc.FN().create()
        a52c2.setType('phi')
        a52c2.setPhiIndex(2)
        a52c2.setScale(Constant(c[3]), -.25)
        a52d2 = SLEPc.FN().create()
        a52d2.setType('phi')
        a52d2.setPhiIndex(3)
        a52d2.setScale(Constant(c[4]), .5)
        a52a_a52b2 = SLEPc.FN().create()
        a52a_a52b2.setType('combine')
        a52a_a52b2.setCombineChildren(SLEPc.FN.CombineType.ADD, a52a2, a52b2)
        a52a_a52b_a52c2 = SLEPc.FN().create()
        a52a_a52b_a52c2.setType('combine')
        a52a_a52b_a52c2.setCombineChildren(SLEPc.FN.CombineType.ADD, a52a_a52b2, a52c2)
        fn_a522 = SLEPc.FN().create()
        fn_a522.setType('combine')
        fn_a522.setCombineChildren(SLEPc.FN.CombineType.ADD, a52a_a52b_a52c2, a52d2)

        a54a = SLEPc.FN().create()
        a54a.setType('phi')
        a54a.setPhiIndex(2)
        a54a.setScale(Constant(c[4]), .25)
        fn_a54 = SLEPc.FN().create()
        fn_a54.setType('combine')
        fn_a54.setCombineChildren(SLEPc.FN.CombineType.ADD, a54a, fn_a522)

        a54a2 = SLEPc.FN().create()
        a54a2.setType('phi')
        a54a2.setPhiIndex(2)
        a54a2.setScale(Constant(c[4]), -.25)
        fn_a542 = SLEPc.FN().create()
        fn_a542.setType('combine')
        fn_a542.setCombineChildren(SLEPc.FN.CombineType.ADD, a54a2, fn_a52)

        a52a3 = SLEPc.FN().create()
        a52a3.setType('phi')
        a52a3.setPhiIndex(2)
        a52a3.setScale(Constant(c[4]), -1)
        a52b3 = SLEPc.FN().create()
        a52b3.setType('phi')
        a52b3.setPhiIndex(3)
        a52b3.setScale(Constant(c[3]), 2)
        a52c3 = SLEPc.FN().create()
        a52c3.setType('phi')
        a52c3.setPhiIndex(2)
        a52c3.setScale(Constant(c[3]), -.5)
        a52d3 = SLEPc.FN().create()
        a52d3.setType('phi')
        a52d3.setPhiIndex(3)
        a52d3.setScale(Constant(c[4]), 1)
        a52a_a52b3 = SLEPc.FN().create()
        a52a_a52b3.setType('combine')
        a52a_a52b3.setCombineChildren(SLEPc.FN.CombineType.ADD, a52a3, a52b3)
        a52a_a52b_a52c3 = SLEPc.FN().create()
        a52a_a52b_a52c3.setType('combine')
        a52a_a52b_a52c3.setCombineChildren(SLEPc.FN.CombineType.ADD, a52a_a52b3, a52c3)
        fn_a523 = SLEPc.FN().create()
        fn_a523.setType('combine')
        fn_a523.setCombineChildren(SLEPc.FN.CombineType.ADD, a52a_a52b_a52c3, a52d3)

        a51a = SLEPc.FN().create()
        a51a.setType('phi')
        a51a.setPhiIndex(1)
        a51a.setScale(Constant(c[4]), .5)
        a51a_2a52 = SLEPc.FN().create()
        a51a_2a52.setType('combine')
        a51a_2a52.setCombineChildren(SLEPc.FN.CombineType.ADD, a51a, fn_a523)
        fn_a51 = SLEPc.FN().create()
        fn_a51.setType('combine')
        fn_a51.setCombineChildren(SLEPc.FN.CombineType.ADD, a51a_2a52, fn_a542)

        A = np.array([[0, 0, 0, 0, 0], [fn_a21, 0, 0, 0, 0], [fn_a31, fn_a32, 0, 0, 0],
                      [fn_a41, fn_a42, fn_a43, 0, 0], [fn_a51, fn_a52, fn_a52, fn_a54, 0]], dtype=object)
        
        b1a = SLEPc.FN().create()
        b1a.setType('phi')
        b1a.setPhiIndex(1)
        b1b = SLEPc.FN().create()
        b1b.setType('phi')
        b1b.setPhiIndex(2)
        b1b.setScale(1, -3)
        b1c = SLEPc.FN().create()
        b1c.setType('phi')
        b1c.setPhiIndex(3)
        b1c.setScale(1, 4)
        b1a_b1b = SLEPc.FN().create()
        b1a_b1b.setType('combine')
        b1a_b1b.setCombineChildren(SLEPc.FN.CombineType.ADD, b1a, b1b)
        fn_b1 = SLEPc.FN().create()
        fn_b1.setType('combine')
        fn_b1.setCombineChildren(SLEPc.FN.CombineType.ADD, b1a_b1b, b1c)

        b4b = SLEPc.FN().create()
        b4b.setType('phi')
        b4b.setPhiIndex(2)
        b4b.setScale(1, -1)
        fn_b4 = SLEPc.FN().create()
        fn_b4.setType('combine')
        fn_b4.setCombineChildren(SLEPc.FN.CombineType.ADD, b1c, b4b)

        b5a = SLEPc.FN().create()
        b5a.setType('phi')
        b5a.setPhiIndex(2)
        b5a.setScale(1, 4)
        b5b = SLEPc.FN().create()
        b5b.setType('phi')
        b5b.setPhiIndex(3)
        b5b.setScale(1, -8)
        fn_b5 = SLEPc.FN().create()
        fn_b5.setType('combine')
        fn_b5.setCombineChildren(SLEPc.FN.CombineType.ADD, b5a, b5b)

        b = np.array([fn_b1, 0, 0, fn_b4, fn_b5], dtype=object)

        super(HOFiveStage, self).__init__(A, b, c)


# mfn functions not reliant on method
def fn2mfn(fn, AA, dt):
    mfn = SLEPc.MFN().create()
    mfn.setOperator(AA)
    fn_alpha, fn_gamma = fn.getScale()
    fn.setScale(-dt * fn_alpha, fn_gamma)
    mfn.setFN(fn)
    return mfn


def apply_petsc(mfn, solver_parameters):
    if mfn != 0:
        opts = PETSc.Options(mfn.getOptionsPrefix())
        for key, value in solver_parameters.items():
            opts[key] = value
        mfn.setFromOptions()
    return


def mfnchi(m, s, solver_parameters):
    mfn_chi = SLEPc.MFN().create()
    mfn_chi.setOperator(m)
    if abs(float(s)) < 1e-15:
        fn_chi = SLEPc.FN().create()
        fn_chi.setType('rational')
        fn_chi.setRationalNumerator(1.0)
    else:    
        fn_chi = SLEPc.FN().create()
        fn_chi.setType('exp')
        fn_chi.setScale(s)
    mfn_chi.setFN(fn_chi)
    gvec = np.vectorize(lambda x: apply_petsc(x, solver_parameters=solver_parameters))
    gvec(mfn_chi)
    return mfn_chi


def mfn_solve(mfn, input, output):
    "mfn * input = output"
    with input.dat.vec_ro as x:
        with output.dat.vec as y:
            mfn.solve(x, y)
    return output


class ExplicitExponentialRKTimeStepper(OptionsManager):

    DEFAULT_MFN_PARAMETERS = {"mfn_type": "krylov",
                              "mfn_tol": 1e-10,
                            #   "mfn_converged_reason": None,
                            #   "mfn_krylov_restart": 0.5,
                              "mfn_ncv": 30}#, "mfn_view": None}
                            #   "mfn_max_it": ,
                            #   "mfn_ncv": ,
                            #    "mfn_monitor"

    def __init__(self, F, g, u, dt, t, method, options_prefix=None, solver_parameters=None):

        self.F = F
        self.g = g
        self.u = u
        self.dt = dt
        self.t = t
        self.mat = MInvK(F)
        self.method = method

        solver_parameters = flatten_parameters(solver_parameters or {})
        for key in self.DEFAULT_MFN_PARAMETERS:
            value = self.DEFAULT_MFN_PARAMETERS[key]
            solver_parameters.setdefault(key, value)
        super().__init__(solver_parameters, options_prefix)

        trial, test = F.arguments()
        self.V = trial.function_space()
        msh = self.V.mesh()

        with self.u.dat.vec_ro as c:
            sizes = c.getSize()
        AI = PETSc.Mat().createPython(
            [sizes, sizes], comm=msh.comm)
        AI.setPythonContext(self.mat)
        AI.setUp()

        self.c = method.c

        self.bMFN = np.array([fn2mfn(fn, AI, dt) for fn in method.b], dtype=object)
        # self.AMFN = np.array([[fn2mfn(fn, AI, dt) for fn in row] for row in method.A], dtype=object)
        self.AMFN = [[(j,fn2mfn(Aij, AI, dt)) for j,Aij in enumerate(row) if Aij != 0] for row in method.A]

        gvec = np.vectorize(lambda x: apply_petsc(x, solver_parameters=solver_parameters))
        for i,row in enumerate(self.AMFN):
            for j,mfnaij in row:
                apply_petsc(mfnaij, solver_parameters=solver_parameters)
        gvec(self.bMFN)

        # un, Un1, Gn1, Un2, Gn2, & un+1
        self.un = self.u
        self.unplus1 = Function(self.V)

        # current stage, gets evaluated and stuffed into g
        self.Uncur = Function(self.V)

        # G applied to all stage values
        self.Gn = [Function(self.V) for _ in range(method.num_stages)]

        self.gconst = Constant(self.c[0])
        self.g = replace(g, {t: t + self.gconst * dt, u: self.Uncur})

        # place to store any bi or aij applied to a Gnj while accumulating into un+1 or Uni
        self.bucket = Function(self.V)

        # assmble all of our SLEPc matrix functions that don't come from the Tableau
        self.mfn_chi_i = [mfnchi(AI, -dt * Constant(self.c[i]), solver_parameters) for i in range(method.num_stages)]
        self.mfn_chi = mfnchi(AI, -dt, solver_parameters)

    def check_convergence(self):
        r"""Check the convergence"""
        for i in [self.AMFN, self.bMFN]:
            for n in i:
                r = n.getConvergedReason()
                try:
                    reason = SLEPc.MFN.ConvergedReason()
                except KeyError:
                    reason = ("unknown reason (petsc4py enum incomplete?), "
                              "try with -MFN_converged_reason")
                if r < 0:
                    raise ConvergenceError(
                        r"""MFN problem failed to converge after %d iterations.
                Reason:
                %s""" % (n.getIterationNumber(), reason)
                    )
                return reason    

    def advance(self):
        # loop over Un and Gn
        for i,row in enumerate(self.AMFN):
            mfn_solve(self.mfn_chi_i[i], self.un, self.Uncur)
            for j,mfnaij in row:
                mfn_solve(mfnaij, self.Gn[j], self.bucket)
                self.Uncur += self.dt * self.bucket
            self.gconst.assign(Constant(self.c[i]))
            self.Gn[i].interpolate(self.g)

        # loop over un+1
        mfn_solve(self.mfn_chi, self.un, self.unplus1)
        for j in range(self.method.num_stages):
            mfn_solve(self.bMFN[j], self.Gn[j], self.bucket)
            self.unplus1 += self.dt * self.bucket
            self.gconst.assign(Constant(self.c[j]))
            self.Gn[j].interpolate(self.g)
        self.u.assign(self.unplus1)


def TwoStageExpRK(Nx, Nt, tfinal):
    # solve problem with exponential Euler from t=0 to tfinal,
    # return L2 error at final time step
    # for t in range (0.0,tfinal):
    t = Constant(0)
    dt = Constant(tfinal / Nt)  # assuming final time is 1
    msh = UnitIntervalMesh(Nx)
    x, = SpatialCoordinate(msh)
    V = FunctionSpace(msh, "CG", 3)
    uu = TrialFunction(V)
    u = Function(V)
    v = TestFunction(V)

    uex = uexact(x, t)
    u.project(uex)

    rhs = expand_time_derivatives(Dt(uex) - div(grad(uex)), t=t, timedep_coeffs=[])
    F = inner(grad(uu), grad(v)) * dx

    c2 = 0.5
    method1 = OneParameterFamily1(c2)

    C = np.array([0.0, 0.5, 0.5, 1.0])
    method2 = CoxAndMatthews2002(C)

    method3 = Krostgrad2005(C)

    C1 = np.array([0.0, 0.5, 0.5, 1.0, 0.5])
    method4 = HOFiveStage(C1)

    stepper = ExplicitExponentialRKTimeStepper(F, rhs, u, dt, t, method4)

    for i in range(Nt):
        stepper.advance()
        t.assign(float(t) + float(dt))

    return errornorm(uex, u) / norm(uex)


Ns = range(2, 6)
errs = [[method(2**n, 2**(n-2), 1.0) for n in Ns] for method in [TwoStageExpRK]] #(2**n, 2**(n-2), 1.0)
for x in errs:
    print(x)