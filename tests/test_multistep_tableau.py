import pytest
from irksome import MultistepMethod
from numpy import allclose, array

# Test some generated multistep methods against known
# tables to make sure the multistep tableaux are right.


def test_Adams_Moulton_zero():
    bt = MultistepMethod('AM', 0)
    a = array([-1.,  1.])
    b = array([0.0,  1.0])
    for (X, Y) in zip([a, b], [bt.a, bt.b]):
        assert allclose(X, Y)


def test_Adams_Moulton():
    bt = MultistepMethod('AM', 4)
    a = array([ 0.,  0.,  0., -1.,  1.])
    b = array([-0.02638889,  0.14722222, -0.36666667,  0.89722222,  0.34861111])
    for (X, Y) in zip([a, b], [bt.a, bt.b]):
        assert allclose(X, Y)


def test_Adams_Bashforth():
    bt = MultistepMethod('AB', 5)
    a = array([ 0.,  0.,  0.,  0., -1.,  1.])
    b = array([ 0.34861111, -1.76944444,  3.63333333, -3.85277778,  2.64027778, 0.])
    for (X, Y) in zip([a, b], [bt.a, bt.b]):
        assert allclose(X, Y)

def test_BDF():
    bt = MultistepMethod('BDF', 5)
    a = array([-0.08759124087591241, 0.5474452554744526, -1.4598540145985401, 2.18978102189781, -2.18978102189781, 1.0])
    b = array([0.0, 0.0, 0.0, 0.0, 0.0, 0.43795620437956206])
    for (X, Y) in zip([a, b], [bt.a, bt.b]):
        assert allclose(X, Y)

@pytest.mark.parametrize('method', tuple([('BDF', k) for k in (1, 2, 3)] + [('AM', k) for k in (0, 2, 5)]))
def test_not_explicit(method):
    bt = MultistepMethod(method[0], method[1])
    assert not bt.is_explicit


@pytest.mark.parametrize('method', [('AB', k) for k in (1, 2, 3)])
def test_explicit(method):
    bt = MultistepMethod(method[0], method[1])
    assert bt.is_explicit
