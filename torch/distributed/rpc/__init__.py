import logging
import threading

from typing import Generator, Tuple
import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


_init_counter = 0
_init_counter_lock = threading.Lock()

def is_available():
    return hasattr(torch._C, "_rpc_init")


if is_available() and not torch._C._rpc_init():
    raise RuntimeError("Failed to initialize torch.distributed.rpc")


if is_available():
    from . import api, backend_registry, functions
    from torch._C._distributed_rpc import (
        _disable_jit_rref_pickle,
        _enable_jit_rref_pickle,
        _disable_server_process_global_profiler,
        _enable_server_process_global_profiler,
        _set_and_start_rpc_agent,
        _reset_current_rpc_agent,
        _delete_all_user_and_unforked_owner_rrefs,
        _destroy_rref_context,
        _set_profiler_node_id,
        _is_current_rpc_agent_set,
        _rref_context_get_debug_info,
        _cleanup_python_rpc_handler,
        _invoke_rpc_builtin,
        _invoke_rpc_python_udf,
        _invoke_rpc_torchscript,
        _invoke_remote_builtin,
        _invoke_remote_python_udf,
        _invoke_remote_torchscript,
        _set_rpc_timeout,
        _get_current_rpc_agent,
        get_rpc_timeout,
        enable_gil_profiling,
        RpcBackendOptions,
        _TensorPipeRpcBackendOptionsBase,
        ProcessGroupRpcBackendOptions,
        RpcAgent,
        PyRRef,
        ProcessGroupAgent,
        TensorPipeAgent,
        RemoteProfilerManager,
        WorkerInfo,
        _DEFAULT_INIT_METHOD,
        _DEFAULT_NUM_SEND_RECV_THREADS,
        _DEFAULT_NUM_WORKER_THREADS,
        _UNSET_RPC_TIMEOUT,
        _DEFAULT_RPC_TIMEOUT_SEC,
    )  # noqa: F401
    from torch._C._distributed_c10d import Store
    from .api import *  # noqa: F401
    from .options import TensorPipeRpcBackendOptions  # noqa: F401
    from .backend_registry import BackendType
    from .server_process_global_profiler import (
        _server_process_global_profile,
    )
    import torch.distributed.autograd as dist_autograd

    import numbers

    rendezvous_iterator: Generator[Tuple[Store, int, int], None, None]

    def init_rpc(
        name,
        backend=None,
        rank=-1,
        world_size=None,
        rpc_backend_options=None,
    ):
        r"""
        Initializes RPC primitives such as the local RPC agent
        and distributed autograd, which immediately makes the current
        process ready to send and receive RPCs.

        Arguments:
            backend (BackendType, optional): The type of RPC backend
                implementation. Supported values include
                ``BackendType.TENSORPIPE`` (the default) and
                ``BackendType.PROCESS_GROUP``. See :ref:`rpc-backends` for more
                information.
            name (str): a globally unique name of this node. (e.g.,
                ``Trainer3``, ``ParameterServer2``, ``Master``, ``Worker1``)
                Name can only contain number, alphabet, underscore, colon,
                and/or dash, and must be shorter than 128 characters.
            rank (int): a globally unique id/rank of this node.
            world_size (int): The number of workers in the group.
            rpc_backend_options (RpcBackendOptions, optional): The options
                passed to the RpcAgent constructor. It must be an agent-specific
                subclass of :class:`~torch.distributed.rpc.RpcBackendOptions`
                and contains agent-specific initialization configurations. By
                default, for all agents, it sets the default timeout to 60
                seconds and performs the rendezvous with an underlying process
                group initialized using ``init_method = "env://"``,
                meaning that environment variables ``MASTER_ADDR`` and
                ``MASTER_PORT`` need to be set properly. See
                :ref:`rpc-backends` for more information and find which options
                are available.
        """

        if backend is not None and not isinstance(backend, backend_registry.BackendType):
            raise TypeError(
                "Argument backend must be a member of BackendType"
            )

        if rpc_backend_options is not None and not isinstance(rpc_backend_options, RpcBackendOptions):
            raise TypeError(
                "Argument rpc_backend_options must be an instance of RpcBackendOptions"
            )

        # To avoid breaking users that passed a ProcessGroupRpcBackendOptions
        # without specifying the backend as PROCESS_GROUP when that was the
        # default, we try to detect the backend from the options when only the
        # latter is passed.
        if backend is None and rpc_backend_options is not None:
            for candidate_backend in BackendType:
                if isinstance(
                    rpc_backend_options,
                    type(
                        backend_registry.construct_rpc_backend_options(
                            candidate_backend
                        )
                    ),
                ):
                    backend = candidate_backend
                    break
            else:
                raise TypeError(
                    f"Could not infer backend for options {rpc_backend_options}"
                )
            # Ignore type error because mypy doesn't handle dynamically generated type objects (#4865)
            if backend != BackendType.TENSORPIPE:  # type: ignore[attr-defined]
                logger.warning(
                    f"RPC was initialized with no explicit backend but with options "  # type: ignore[attr-defined]
                    f"corresponding to {backend}, hence that backend will be used "
                    f"instead of the default {BackendType.TENSORPIPE}. To silence this "
                    f"warning pass `backend={backend}` explicitly."
                )

        if backend is None:
            backend = BackendType.TENSORPIPE  # type: ignore[attr-defined]

        if backend == BackendType.PROCESS_GROUP:  # type: ignore[attr-defined]
            logger.warning(
                "RPC was initialized with the PROCESS_GROUP backend which is "
                "deprecated and slated to be removed and superseded by the TENSORPIPE "
                "backend. It is recommended to migrate to the TENSORPIPE backend."
            )

        if rpc_backend_options is None:
            # default construct a set of RPC backend options.
            rpc_backend_options = backend_registry.construct_rpc_backend_options(
                backend
            )

        # Rendezvous.
        # This rendezvous state sometimes is destroyed before all processes
        # finishing handshaking. To avoid that issue, we make it global to
        # keep it alive.
        global rendezvous_iterator
        rendezvous_iterator = torch.distributed.rendezvous(
            rpc_backend_options.init_method, rank=rank, world_size=world_size
        )
        store, _, _ = next(rendezvous_iterator)

        # Use a PrefixStore to distinguish multiple invocations.
        with _init_counter_lock:
            global _init_counter
            store = dist.PrefixStore(str('rpc_prefix_{}'.format(_init_counter)), store)
            _init_counter += 1

        # Initialize autograd before RPC since _init_rpc_backend guarantees all
        # processes sync via the store. If we initialize autograd after RPC,
        # there could be a race where some nodes might have initialized autograd
        # and others might not have. As a result, a node calling
        # torch.distributed.autograd.backward() would run into errors since
        # other nodes might not have been initialized.
        dist_autograd._init(rank)

        _set_profiler_node_id(rank)
        # Initialize RPC.
        _init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)


    def _validate_rpc_args(backend, store, name, rank, world_size, rpc_backend_options):
        type_mapping = {
            backend: backend_registry.BackendType,
            store: dist.Store,
            name: str,
            rank: numbers.Integral,
            world_size: numbers.Integral,
            rpc_backend_options: RpcBackendOptions,
        }
        for arg, arg_type in type_mapping.items():
            if not isinstance(arg, arg_type):
                raise RuntimeError(
                    "Argument {} must be of type {} but got type {}".format(
                        arg, arg_type, type(arg)
                    )
                )


    def _init_rpc_backend(
        backend=BackendType.TENSORPIPE,  # type: ignore[attr-defined]
        store=None,
        name=None,
        rank=-1,
        world_size=-1,
        rpc_backend_options=None,
    ):

        _validate_rpc_args(backend, store, name, rank, world_size, rpc_backend_options)

        if _is_current_rpc_agent_set():
            raise RuntimeError("RPC is already initialized")

        # Initialize RPC.
        rpc_agent = backend_registry.init_backend(
            backend,
            store=store,
            name=name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        api._init_rpc_states(rpc_agent)


    @api._require_initialized
    def _get_debug_info():
        info = _rref_context_get_debug_info()
        info.update(api._get_current_rpc_agent().get_debug_info())
        info.update(dist_autograd._get_debug_info())
        return info
