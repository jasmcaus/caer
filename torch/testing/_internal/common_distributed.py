
from multiprocessing import Manager
from contextlib import contextmanager
from io import StringIO
import os
import sys
import tempfile
import time
import unittest
import logging
import traceback
import types

from typing import NamedTuple
from functools import wraps

import torch
import torch.distributed as c10d

from functools import partial, reduce
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, FILE_SCHEMA

class TestSkip(NamedTuple):
    exit_code: int
    message: str


TEST_SKIPS = {
    "backend_unavailable": TestSkip(72, "Skipped because distributed backend is not available."),
    "small_worldsize": TestSkip(73, "Skipped due to small world size."),
    "no_cuda": TestSkip(74, "CUDA is not available."),
    "multi-gpu": TestSkip(75, "Need at least 2 CUDA devices"),
    "nccl": TestSkip(76, "c10d not compiled with NCCL support"),
    "skipIfRocm": TestSkip(78, "Test skipped for ROCm")
}

def skip_if_no_gpu(func):
    """ Nccl multigpu tests require at least 2 GPUS. Skip if this is not met"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            sys.exit(TEST_SKIPS["no_cuda"].exit_code)
        if torch.cuda.device_count() < int(os.environ["WORLD_SIZE"]):
            message = "Need at least {} CUDA devices".format(os.environ["WORLD_SIZE"])
            TEST_SKIPS["multi-gpu"] = TestSkip(75, message)
            sys.exit(TEST_SKIPS["multi-gpu"].exit_code)

        return func(*args, **kwargs)

    return wrapper


def skip_if_small_worldsize(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if (os.environ["BACKEND"] != "mpi") and int(os.environ["WORLD_SIZE"]) <= 2:
            sys.exit(TEST_SKIPS["small_worldsize"].exit_code)

        return func(*args, **kwargs)

    return wrapper


def skip_if_not_multigpu(func):
    """Multi-GPU tests requires at least 2 GPUS. Skip if this is not met."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
                return func(*args, **kwargs)
            message = "Need at least {} CUDA devices".format(2)
            TEST_SKIPS["multi-gpu"] = TestSkip(75, message)
            sys.exit(TEST_SKIPS['multi-gpu'].exit_code)
        return wrapper

    return decorator

def require_n_gpus_for_nccl_backend(n, backend):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if backend == "nccl" and torch.cuda.device_count() < n:
                message = "Need at least {} CUDA devices".format(n)
                TEST_SKIPS["multi-gpu"] = TestSkip(75, message)
                sys.exit(TEST_SKIPS['multi-gpu'].exit_code)
            else:
                return func(*args, **kwargs)
        return wrapper

    return decorator

def skip_if_lt_x_gpu(x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            message = "Need at least {} CUDA devices".format(x)
            TEST_SKIPS["multi-gpu"] = TestSkip(75, message)
            sys.exit(TEST_SKIPS['multi-gpu'].exit_code)
        return wrapper

    return decorator

def requires_gloo():
    return unittest.skipUnless(
        c10d.is_gloo_available(),
        "c10d was not compiled with the Gloo backend",
    )

def requires_nccl_version(version, msg):
    if not c10d.is_nccl_available():
        return unittest.skip(
            "c10d was not compiled with the NCCL backend",
        )
    else:
        return unittest.skipIf(
            torch.cuda.nccl.version() < version,
            "Requires NCCL version greater than or equal to: {}, found: {}, reason: {}".format(
                version,
                torch.cuda.nccl.version(), msg),
        )

def requires_nccl():
    return unittest.skipUnless(
        c10d.is_nccl_available(),
        "c10d was not compiled with the NCCL backend",
    )


def requires_mpi():
    return unittest.skipUnless(
        c10d.is_mpi_available(),
        "c10d was not compiled with the MPI backend",
    )

def skip_if_rocm_single_process(func):
    """Skips a test for ROCm in a single process environment"""
    func.skip_if_rocm = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TEST_WITH_ROCM:
            return func(*args, **kwargs)
        raise unittest.SkipTest("Test skipped for ROCm")

    return wrapper

def skip_if_rocm(func):
    """Skips a test for ROCm"""
    func.skip_if_rocm = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TEST_WITH_ROCM:
            return func(*args, **kwargs)
        sys.exit(TEST_SKIPS['skipIfRocm'].exit_code)

    return wrapper

def skip_if_win32():
    return unittest.skipIf(
        sys.platform == 'win32',
        "This unit test case is not supportted on Windows platform",
    )

TIMEOUT_DEFAULT = 100
TIMEOUT_OVERRIDE = {"test_ddp_uneven_inputs": 400}


def create_device(interface=None):
    if sys.platform == 'win32' or interface is None:
        return c10d.ProcessGroupGloo.create_device(hostname="127.0.0.1")
    else:
        return c10d.ProcessGroupGloo.create_device(interface=interface)


def get_timeout(test_id):
    return TIMEOUT_OVERRIDE.get(test_id.split('.')[-1], TIMEOUT_DEFAULT)

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def simple_sparse_reduce_tests(rank, world_size, num_inputs=1):
    """
    Generate a number of basic test cases for sparse reduction.
    These cover tensors with a varying number of sparse dimensions and a varying
    number of dense dimensions. The only reduction operation we support is sum.
    """
    def generate(rank, world_size, sparse_dims=1, dense_dims=0):
        # First sparse dimension is [0..rank].
        # Subsequent dimensions are always 0, so we know there is
        # a non-empty intersection between any two sparse tensors.
        indices = torch.reshape(torch.arange(rank + 1), (1, rank + 1))
        shape = [world_size] + [2 for _ in range(dense_dims)]
        for _ in range(sparse_dims - 1):
            indices = torch.cat((indices, torch.zeros(1, rank + 1)))
            shape.append(world_size)
        values = torch.ones([rank + 1] + [2 for _ in range(dense_dims)])
        return torch.sparse_coo_tensor(indices, values, shape)

    def compute_sum(fn, world_size):
        return reduce(lambda a, b: a + b, [fn(rank, world_size) for rank in range(world_size)])

    return [
        (
            [
                fn(num_inputs * rank + i, num_inputs * world_size)
                for i in range(num_inputs)
            ],
            [
                compute_sum(fn, num_inputs * world_size)
                for i in range(num_inputs)
            ],
        )
        for fn in [
            partial(generate, sparse_dims=1),
            partial(generate, sparse_dims=2),
            partial(generate, sparse_dims=3),
            partial(generate, dense_dims=1),
            partial(generate, dense_dims=2),
            partial(generate, dense_dims=3),
        ]
    ]

tmp_dir = None
def initialize_temp_directories(init_method=None):
    global tmp_dir
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ["TEMP_DIR"] = tmp_dir.name
    os.mkdir(os.path.join(tmp_dir.name, "barrier"))
    os.mkdir(os.path.join(tmp_dir.name, "test_dir"))
    init_dir_path = os.path.join(tmp_dir.name, "init_dir")
    os.mkdir(init_dir_path)
    # Set init method if specified.
    if init_method is not None:
        os.environ["INIT_METHOD"] = init_method
    else:
        os.environ["INIT_METHOD"] = FILE_SCHEMA + os.path.join(
            init_dir_path, "shared_init_file"
        )

def cleanup_temp_dir():
    if tmp_dir is not None:
        tmp_dir.cleanup()

# [How does MultiProcessTestCase work?]
# Each MultiProcessTestCase instance uses 1 + `world_size()` processes, by
# default `world_size()` returns 4. Let's take `test_rpc_spawn.py` as an
# example which inherits from this class. Its `Setup()` methods calls into
# `MultiProcessTestCase._spawn_processes()` which spawns `world_size()`
# subprocesses. During the spawn, the main process passes the test name to
# subprocesses, and the name is acquired from self.id(). The subprocesses
# then use the provided test function name to retrieve the function attribute
# from the test instance and run it. The main process simply waits for all
# subprocesses to join.
class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1
    # This exit code is used to indicate that the test code had an error and
    # exited abnormally. There are certain tests that might use sys.exit() to
    # simulate failures and in those cases, we can't have an exit code of 0,
    # but we still want to ensure we didn't run into any other errors.
    TEST_ERROR_EXIT_CODE = 10

    @property
    def world_size(self):
        return 4

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                try:
                    fn()
                except Exception as e:
                    logging.error('Caught exception: \n{}exiting process with exit code: {}'
                                  .format(traceback.format_exc(), MultiProcessTestCase.TEST_ERROR_EXIT_CODE))
                    sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        return types.MethodType(wrapper, self)

    # The main process spawns N subprocesses that run the test.
    # Constructor patches current instance test method to
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def setUp(self):
        super().setUp()
        self.skip_return_code_checks = []
        self.processes = []
        self.rank = self.MAIN_PROCESS_RANK
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        global TEST_SKIPS
        self.old_test_skips = TEST_SKIPS.copy()

    def tearDown(self):
        super().tearDown()
        for p in self.processes:
            p.terminate()
        # Each Process instance holds a few open file descriptors. The unittest
        # runner creates a new TestCase instance for each test method and keeps
        # it alive until the end of the entire suite. We must thus reset the
        # processes to prevent an effective file descriptor leak.
        self.processes = []

    def _current_test_name(self):
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    def _start_processes(self, proc):
        test_skips_manager = Manager()
        test_skips = test_skips_manager.dict()
        global TEST_SKIPS
        test_skips.update(TEST_SKIPS)
        TEST_SKIPS = test_skips

        self.processes = []
        for rank in range(int(self.world_size)):
            process = proc(
                target=self.__class__._run,
                name='process ' + str(rank),
                args=(rank, self._current_test_name(), self.file_name))
            process.start()
            self.processes.append(process)

    def _fork_processes(self):
        proc = torch.multiprocessing.get_context("fork").Process
        self._start_processes(proc)

    def _spawn_processes(self):
        proc = torch.multiprocessing.get_context("spawn").Process
        self._start_processes(proc)

    @classmethod
    def _run(cls, rank, test_name, file_name):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retrieving a corresponding test and executing it.
        getattr(self, test_name)()
        # exit to avoid run teardown() for fork processes
        sys.exit(0)

    def _join_processes(self, fn):
        timeout = get_timeout(self.id())
        start_time = time.time()
        subprocess_error = False
        try:
            while True:
                # check to see if any subprocess exited with an error early.
                for (i, p) in enumerate(self.processes):
                    # This is the exit code processes exit with if they
                    # encountered an exception.
                    if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                        print("Process {} terminated with exit code {}, terminating remaining processes.".format(i, p.exitcode))
                        active_children = torch.multiprocessing.active_children()
                        for ac in active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break
                # All processes have joined cleanly if they all a valid exitcode
                if all([p.exitcode is not None for p in self.processes]):
                    break
                # Check if we should time out the test. If so, we terminate each process.
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print(
                        "Timing out after {} seconds and killing subprocesses.".format(
                            timeout
                        )
                    )
                    for p in self.processes:
                        p.terminate()
                    break
                # Sleep to avoid excessive busy polling.
                time.sleep(0.1)
            elapsed_time = time.time() - start_time
            if fn in self.skip_return_code_checks:
                self._check_no_test_errors(elapsed_time)
            else:
                self._check_return_codes(elapsed_time)
        finally:
            global TEST_SKIPS
            TEST_SKIPS = self.old_test_skips

    def _check_no_test_errors(self, elapsed_time):
        """
        Checks that we didn't have any errors thrown in the child processes.
        """
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError('Process {} timed out after {} seconds'.format(i, elapsed_time))
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    def _check_return_codes(self, elapsed_time):
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        first_process = self.processes[0]
        # first, we check if there are errors in actual processes
        # (via TEST_ERROR_EXIT CODE), and raise an exception for those.
        # the reason we do this is to attempt to raise a more helpful error
        # message than "Process x terminated/timed out"
        # TODO: we should pipe the exception of the failed subprocess here.
        # Currently, the actual exception is displayed as a logging output.
        errored_processes = [
            (i, p)
            for i, p in enumerate(self.processes)
            if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE
        ]
        if errored_processes:
            error = "Processes {} exited with error code {}".format(
                " ".join([str(i) for (i, _) in errored_processes]),
                MultiProcessTestCase.TEST_ERROR_EXIT_CODE,
            )
            raise RuntimeError(error)
        # If no process exited uncleanly, we check for timeouts, and then ensure
        # each process exited cleanly.
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError('Process {} terminated or timed out after {} seconds'.format(i, elapsed_time))
            self.assertEqual(
                p.exitcode,
                first_process.exitcode,
                msg="Expect process {} exit code to match Process 0 exit code of {}, but got {}".format(
                    i, first_process.exitcode, p.exitcode
                ),
            )
        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                raise unittest.SkipTest(skip.message)
        self.assertEqual(
            first_process.exitcode,
            0,
            msg="Expected zero exit code but got {}".format(first_process.exitcode)
        )

    @property
    def is_master(self):
        return self.rank == 0
