import pytest
from irksome import GaussLegendre, LobattoIIIA, LobattoIIIC, QinZhang, RadauIIA
from numpy import allclose, array, sqrt

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
    Aexplicit = array([[-1/12, 5/12], [-3/4, 7/4]])
    b = array([3/4, 1/4])
    c = array([1/3, 1])

    for (X, Y) in zip([A, Aexplicit, b, c], [bt.A, bt.Aexplicit, bt.b, bt.c]):
        assert allclose(X, Y)


@pytest.mark.parametrize('bt', tuple([RadauIIA(k) for k in (1, 2, 3)]
                                     + [LobattoIIIC(k) for k in (2, 3)]))
def test_is_stiffly_accurate(bt):
    assert bt.is_stiffly_accurate


@pytest.mark.parametrize('bt', tuple([GaussLegendre(k) for k in (1, 2, 3)]
                                     + [QinZhang()]))
def test_is_not_stiffly_accurate(bt):
    assert not bt.is_stiffly_accurate
