from firedrake import *
from firedrake.adjoint import *
import pytest


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    if annotate_tape():
        pause_annotation()


def test_adjoint_subfunctions():
    from networkx.algorithms import is_weakly_connected

    mesh = UnitIntervalMesh(8)
    R = FunctionSpace(mesh, "R", 0)

    kappa = Function(R).assign(2.0)
    u = Function(R)
    usub = u.subfunctions[0]

    print(f"{str(kappa) = } | {str(u) = } | {str(usub) = }")

    with set_working_tape() as tape:

        u.assign(kappa)

        # This line passes
        # u *= 2

        # This line fails
        usub *= 2

        J = assemble(inner(u, u) * dx)
        rf = ReducedFunctional(J, Control(kappa), tape=tape)
        assert is_weakly_connected(tape.create_graph())
    rf.tape.visualise_pdf("tape.pdf")

    rf.derivative()
    assert taylor_test(rf, kappa, Constant(0.01)) > 1.9


if __name__ == '__main__':
    continue_annotation()
    test_adjoint_subfunctions()
    pause_annotation()
