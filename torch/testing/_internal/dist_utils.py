
import time
from functools import partial, wraps
import re
import sys

import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import _rref_context_get_debug_info  # type: ignore[attr-defined]
from torch.testing._internal.common_utils import FILE_SCHEMA


if not dist.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)


INIT_METHOD_TEMPLATE = FILE_SCHEMA + "{file_name}"


def single_threaded_process_group_agent(f):
    """
    Forces ProcessGroupAgent to use only a single thread in the ThreadPool for
    sending and processing requests.
    """
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        backend_type = self.rpc_backend
        if backend_type == rpc.backend_registry.BackendType["PROCESS_GROUP"]:
            self.rpc_backend_options = rpc.backend_registry.construct_rpc_backend_options(
                self.rpc_backend,
                init_method=self.init_method,
                num_send_recv_threads=1,
            )
        return_value = f(self, *args, **kwargs)
        return return_value
    return wrapper


def dist_init(old_test_method=None, setup_rpc=True, clean_shutdown=True,
              faulty_messages=None, messages_to_delay=None):
    """
    We use this decorator for setting up and tearing down state since
    MultiProcessTestCase runs each `test*` method in a separate process and
    each process just runs the `test*` method without actually calling
    'setUp' and 'tearDown' methods of unittest.

    Note: pass the string representation of MessageTypes that should be used
    with the faulty agent's send function. By default, all retriable messages
    ("RREF_FORK_REQUEST", "RREF_CHILD_ACCEPT", "RREF_USER_DELETE",
    "CLEANUP_AUTOGRAD_CONTEXT_REQ") will use the faulty send (this default is
    set from faulty_rpc_agent_test_fixture.py).
    """

    # If we use dist_init without arguments (ex: @dist_init), old_test_method is
    # appropriately set and we return the wrapper appropriately. On the other
    # hand if dist_init has arguments (ex: @dist_init(clean_shutdown=False)),
    # old_test_method is None and we return a functools.partial which is the real
    # decorator that is used and as a result we recursively call dist_init with
    # old_test_method and the rest of the arguments appropriately set.
    if old_test_method is None:
        return partial(
            dist_init,
            setup_rpc=setup_rpc,
            clean_shutdown=clean_shutdown,
            faulty_messages=faulty_messages,
            messages_to_delay=messages_to_delay,
        )

    @wraps(old_test_method)
    def new_test_method(self, *arg, **kwargs):
        # Setting _ignore_rref_leak to make sure OwnerRRefs are properly deleted
        # in tests.
        import torch.distributed.rpc.api as api
        api._ignore_rref_leak = False

        self.worker_id = self.rank

        self.setup_fault_injection(faulty_messages, messages_to_delay)

        if setup_rpc:
            rpc.init_rpc(
                name="worker%d" % self.rank,
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )

        return_value = old_test_method(self, *arg, **kwargs)

        if setup_rpc:
            rpc.shutdown(graceful=clean_shutdown)

        return return_value

    return new_test_method


def noop():
    pass

def wait_until_node_failure(rank, expected_error_regex=".*"):
    '''
    Loops until an RPC to the given rank fails. This is used to
    indicate that the node has failed in unit tests.
    Args:
    rank (int): Rank of the node expected to fail
    expected_error_regex (optional, str): Regex of exception message expected. Useful to ensure a specific failure
    occurs, not just any.
    '''
    while True:
        try:
            rpc.rpc_sync("worker{}".format(rank), noop, args=())
            time.sleep(0.1)
        except Exception as e:
            if re.search(pattern=expected_error_regex, string=str(e)):
                return str(e)


def wait_until_pending_futures_and_users_flushed(timeout=20):
    '''
    The RRef protocol holds forkIds of rrefs in a map until those forks are
    confirmed by the owner. The message confirming the fork may arrive after
    our tests check whether this map is empty, which leads to failures and
    flaky tests. to_here also does not guarantee that we have finished
    processind the owner's confirmation message for the RRef. This function
    loops until the map is empty, which means the messages have been received
    as processed. Call this function before asserting the map returned by
    _get_debug_info is empty.
    '''
    start = time.time()
    while True:
        debug_info = _rref_context_get_debug_info()
        num_pending_futures = int(debug_info["num_pending_futures"])
        num_pending_users = int(debug_info["num_pending_users"])
        if num_pending_futures == 0 and num_pending_users == 0:
            break
        time.sleep(0.1)
        if time.time() - start > timeout:
            raise ValueError(
                "Timed out waiting to flush pending futures and users, had {} pending futures and {} pending users".format(
                    num_pending_futures, num_pending_users
                )
            )


def get_num_owners_and_forks():
    """
    Retrieves number of OwnerRRefs and forks on this node from
    _rref_context_get_debug_info.
    """
    rref_dbg_info = _rref_context_get_debug_info()
    num_owners = rref_dbg_info["num_owner_rrefs"]
    num_forks = rref_dbg_info["num_forks"]
    return num_owners, num_forks


def wait_until_owners_and_forks_on_rank(num_owners, num_forks, rank, timeout=20):
    """
    Waits until timeout for num_forks and num_owners to exist on the rank. Used
    to ensure proper deletion of RRefs in tests.
    """
    start = time.time()
    while True:
        num_owners_on_rank, num_forks_on_rank = rpc.rpc_sync(
            worker_name(rank), get_num_owners_and_forks, args=(), timeout=5
        )
        num_owners_on_rank = int(num_owners_on_rank)
        num_forks_on_rank = int(num_forks_on_rank)
        if num_owners_on_rank == num_owners and num_forks_on_rank == num_forks:
            return
        time.sleep(1)
        if time.time() - start > timeout:
            raise ValueError(
                "Timed out waiting {} sec for {} owners and {} forks on rank, had {} owners and {} forks".format(
                    timeout, num_owners, num_forks, num_owners_on_rank, num_forks_on_rank
                )
            )


def initialize_pg(init_method, rank, world_size):
    # This is for tests using `dist.barrier`.
    # For `RpcAgent` other than `ProcessGroupAgent`,
    # no `_default_pg` is initialized.
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

def worker_name(rank):
    return "worker{}".format(rank)

def get_function_event(function_events, partial_event_name):
    """
    Returns the first event that matches partial_event_name in the provided
    function_events. These function_events should be the output of
    torch.autograd.profiler.function_events().

    Args:
    function_events: function_events returned by the profiler.
    event_name (str): partial key that the event was profiled with.
    """
    event = [event for event in function_events if partial_event_name in event.name][0]
    return event
