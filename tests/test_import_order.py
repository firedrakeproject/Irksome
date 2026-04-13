import pytest


def test_wrong_import_order():
    from firedrake import UnitIntervalMesh, assemble, dx
    mesh = UnitIntervalMesh(2)
    assemble(1*dx(domain=mesh))

    # the best we can do is to check for Exception
    # because we cannot import IrksomeImportOrderException
    # after firedrake without raising the exception
    # or before firedrake without preventing the failure.
    with pytest.raises(Exception):
        import irksome  # noqa F401

    with pytest.raises(Exception):
        from irksome import Dt  # noqa F401
