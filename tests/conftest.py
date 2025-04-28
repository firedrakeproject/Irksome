"""Global test configuration."""

import gc
import os
import pytest

from subprocess import check_call
from mpi4py import MPI


@pytest.fixture(scope="module", autouse=True)
def check_empty_tape(request):
    """Check that the tape is empty at the end of each module"""
    from pyadjoint.tape import annotate_tape, get_working_tape

    def fin():
        # make sure taping is switched off
        assert not annotate_tape()

        # make sure the tape is empty
        tape = get_working_tape()
        if tape is not None:
            assert len(tape.get_blocks()) == 0

    request.addfinalizer(fin)


@pytest.fixture(autouse=True)
def disable_gc_on_parallel(request):
    """ Disables garbage collection on parallel tests,
    but only when run on Jenkins CI
    """
    if (MPI.COMM_WORLD.size > 1) and ("FIREDRAKE_CI_TESTS" in os.environ):
        gc.disable()
        assert not gc.isenabled()
        request.addfinalizer(restart_gc)


def restart_gc():
    """ Finaliser for restarting garbage collection
    """
    gc.enable()
    assert gc.isenabled()


def parallel(item):
    """Run a test in parallel.

    :arg item: The test item to run.
    """
    if MPI.COMM_WORLD.size > 1:
        raise RuntimeError("parallel test can't be run within parallel environment")
    marker = item.get_closest_marker("parallel")
    if marker is None:
        raise RuntimeError("Parallel test doesn't have parallel marker")
    nprocs = marker.kwargs.get("nprocs", 3)
    if nprocs < 2:
        raise RuntimeError("Need at least two processes to run parallel test")

    # Only spew tracebacks on rank 0.
    # Run xfailing tests to ensure that errors are reported to calling process
    call = ["mpiexec", "-n", "1", "python", "-m", "pytest", "--runxfail", "-s", "-q", "%s::%s" % (item.fspath, item.name)]
    call.extend([":", "-n", "%d" % (nprocs - 1), "python", "-m", "pytest", "--runxfail", "--tb=no", "-q",
                 "%s::%s" % (item.fspath, item.name)])
    check_call(call)


def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "parallel(nprocs): mark test to run in parallel on nprocs processors")


def pytest_runtest_setup(item):
    if item.get_closest_marker("parallel"):
        if MPI.COMM_WORLD.size > 1:
            # Turn on source hash checking
            from firedrake import parameters
            from functools import partial

            def _reset(check):
                parameters["pyop2_options"]["check_src_hashes"] = check

            # Reset to current value when test is cleaned up
            item.addfinalizer(partial(_reset,
                                      parameters["pyop2_options"]["check_src_hashes"]))

            parameters["pyop2_options"]["check_src_hashes"] = True
        else:
            # Blow away function arg in "master" process, to ensure
            # this test isn't run on only one process.
            item.obj = lambda *args, **kwargs: True


def pytest_runtest_call(item):
    if item.get_closest_marker("parallel") and MPI.COMM_WORLD.size == 1:
        # Spawn parallel processes to run test
        parallel(item)
