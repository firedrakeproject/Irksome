import numpy
from .ButcherTableaux import ButcherTableau


mri_gark_dict = {
    "ERK22a": (
        [1/2, 1/2],
        [[[1/2, 0],
          [-1/2, 1]]],
        2),
    "ERK33a": (
        [1/3, 1/3, 1/3],
        [[[1/3, 0, 0],
          [-1/3, 2/3, 0],
          [0, -2/3, 1]],
         [[0, 0, 0],
          [0, 0, 0],
          [1/2, 0, -1/2]]],
        3),
    "ERK45a": (
        [1/5, 1/5, 1/5, 1/5, 1/5],
        [[[1/5, 0, 0, 0, 0],
          [-53/16, 281/80, 0, 0, 0],
          [-36562993/71394880, 34903117/17848720, -88770499/71394880, 0, 0],
          [-7631593/71394880, -166232021/35697440, 6068517/1519040, 8644289/8924360, 0],
          [277061/303808, -209323/1139280, -1360217/1139280, -148789/56964, 147889/45120]],
         [[0, 0, 0, 0, 0],
          [503/80, -503/80, 0, 0, 0],
          [-1365537/35697440, 4963773/7139488, -1465833/2231090, 0, 0],
          [66974357/35697440, 21445367/7139488, -3, -8388609/4462180, 0],
          [-18227/7520, 2, 1, 5, -41933/7520]]],
        4),
}


class MRIGARK(ButcherTableau):
    """Explicit multirate infinitesimal GARK method of Sandu (2019).

    Each slow stage is followed by a fast integration over ``dc[i] * dt``
    driven by a polynomial-in-time combination of the slow tendencies.
    The inherited ``A``, ``b``, ``c`` describe the slow method obtained
    when the fast part vanishes.

    :arg name: one of ``"ERK22a"``, ``"ERK33a"``, ``"ERK45a"``.
    """
    def __init__(self, name):
        try:
            dc, Gamma, order = mri_gark_dict[name]
        except KeyError:
            raise NotImplementedError(f"No MRI-GARK method named {name}")
        self.name = name
        self.dc = numpy.array(dc)
        assert all(self.dc > 0)
        self.Gamma = [numpy.array(G) for G in Gamma]
        s = len(dc)
        gbar = sum(G / (k + 1) for k, G in enumerate(self.Gamma))
        A = numpy.array([gbar[:i].sum(axis=0) for i in range(s)])
        b = gbar.sum(axis=0)
        c = numpy.concatenate(([0.0], numpy.cumsum(self.dc)[:-1]))
        super().__init__(A, b, None, c, order, None, None)

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"
