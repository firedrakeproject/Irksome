from numpy import sqrt, array, allclose
from irksome import GaussLegendre, LobattoIIIA, LobattoIIIC, RadauIIA

# Test some generated collocation methods against known
# tables to make sure the Butcher tableaux are right.


def test_GaussLegendre():
    bt = GaussLegendre(2)
    A = array([[1 / 4, 1 / 4 - sqrt(3) / 6],
               [1 / 4 + sqrt(3) / 6, 1 / 4]])
    b = array([1 / 2, 1 / 2])
    # btilde = array([1 / 2 + sqrt(3) / 6, 1 / 2 - sqrt(3) / 6])
    c = array([1 / 2 - sqrt(3) / 6, 1 / 2 + sqrt(3) / (6)])
    # assert allclose(btilde, bt.btilde)
    for (X, Y) in zip([A, b, c], [bt.A, bt.b, bt.c]):
        assert allclose(X, Y)


def test_LobattoIIIA():
    bt = LobattoIIIA(3)
    A = array([[0, 0, 0], [5/24, 1/3, -1/24], [1/6, 2/3, 1/6]])
    b = array([1/6, 2/3, 1/6])
    c = array([0, 1/2, 1])

    for (X, Y) in zip([A, b, c], [bt.A, bt.b, bt.c]):
        assert allclose(X, Y)


def test_LobattoIIIC():
    bt = LobattoIIIC(3)
    A = array([[1/6, -1/3, 1/6], [1/6, 5/12, -1/12], [1/6, 2/3, 1/6]])
    b = array([1/6, 2/3, 1/6])
    c = array([0, 1/2, 1])

    for (X, Y) in zip([A, b, c], [bt.A, bt.b, bt.c]):
        assert allclose(X, Y)


def test_RadauIIA():
    bt = RadauIIA(2)
    A = array([[5/12, -1/12], [3/4, 1/4]])
    b = array([3/4, 1/4])
    c = array([1/3, 1])

    for (X, Y) in zip([A, b, c], [bt.A, bt.b, bt.c]):
        assert allclose(X, Y)
