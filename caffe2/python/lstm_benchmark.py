## @package lstm_benchmark
# Module caffe2.python.lstm_benchmark





from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, utils, rnn_cell, model_helper
from caffe2.python import recurrent

import argparse
import numpy as np
import time

import logging

logging.basicConfig()
log = logging.getLogger("lstm_bench")
log.setLevel(logging.DEBUG)


def generate_data(T, shape, num_labels, fixed_shape):
    '''
    Fill a queue with input data
    '''
    log.info("Generating T={} sequence batches".format(T))

    generate_input_init_net = core.Net('generate_input_init')
    queue = generate_input_init_net.CreateBlobsQueue(
        [], "inputqueue", num_blobs=1, capacity=T,
    )
    label_queue = generate_input_init_net.CreateBlobsQueue(
        [], "labelqueue", num_blobs=1, capacity=T,
    )

    workspace.RunNetOnce(generate_input_init_net)
    generate_input_net = core.Net('generate_input')

    generate_input_net.EnqueueBlobs([queue, "scratch"], ["scratch"])
    generate_input_net.EnqueueBlobs([label_queue, "label_scr"], ["label_scr"])
    np.random.seed(2603)

    entry_counts = []
    for t in range(T):
        if (t % (max(10, T // 10)) == 0):
            print("Generating data {}/{}".format(t, T))
        # Randomize the seqlength
        random_shape = (
            [np.random.randint(1, shape[0])] + shape[1:]
            if t > 0 and not fixed_shape else shape
        )
        X = np.random.rand(*random_shape).astype(np.float32)
        batch_size = random_shape[1]
        L = num_labels * batch_size
        labels = (np.random.rand(random_shape[0]) * L).astype(np.int32)
        workspace.FeedBlob("scratch", X)
        workspace.FeedBlob("label_scr", labels)
        workspace.RunNetOnce(generate_input_net.Proto())
        entry_counts.append(random_shape[0] * random_shape[1])

    log.info("Finished data generation")

    return queue, label_queue, entry_counts


def create_model(args, queue, label_queue, input_shape):
    model = model_helper.ModelHelper(name="LSTM_bench")
    seq_lengths, target = \
        model.net.AddExternalInputs(
            'seq_lengths',
            'target',
        )

    input_blob = model.net.DequeueBlobs(queue, "input_data")
    labels = model.net.DequeueBlobs(label_queue, "label")

    init_blobs = []
    if args.implementation in ["own", "static", "static_dag"]:
        T = None
        if "static" in args.implementation:
            assert args.fixed_shape, \
                "Random input length is not static RNN compatible"
            T = args.seq_length
            print("Using static RNN of size {}".format(T))

        for i in range(args.num_layers):
            hidden_init, cell_init = model.net.AddExternalInputs(
                "hidden_init_{}".format(i),
                "cell_init_{}".format(i)
            )
            init_blobs.extend([hidden_init, cell_init])

        output, last_hidden, _, last_state = rnn_cell.LSTM(
            model=model,
            input_blob=input_blob,
            seq_lengths=seq_lengths,
            initial_states=init_blobs,
            dim_in=args.input_dim,
            dim_out=[args.hidden_dim] * args.num_layers,
            scope="lstm1",
            memory_optimization=args.memory_optimization,
            forward_only=args.forward_only,
            drop_states=True,
            return_last_layer_only=True,
            static_rnn_unroll_size=T,
        )

        if "dag" in args.implementation:
            print("Using DAG net type")
            model.net.Proto().type = 'dag'
            model.net.Proto().num_workers = 4

    elif args.implementation == "cudnn":
        # We need to feed a placeholder input so that RecurrentInitOp
        # can infer the dimensions.
        init_blobs = model.net.AddExternalInputs("hidden_init", "cell_init")
        model.param_init_net.ConstantFill([], input_blob, shape=input_shape)
        output, last_hidden, _ = rnn_cell.cudnn_LSTM(
            model=model,
            input_blob=input_blob,
            initial_states=init_blobs,
            dim_in=args.input_dim,
            dim_out=args.hidden_dim,
            scope="cudnnlstm",
            num_layers=args.num_layers,
        )

    else:
        assert False, "Unknown implementation"

    weights = model.net.UniformFill(labels, "weights")
    softmax, loss = model.net.SoftmaxWithLoss(
        [model.Flatten(output), labels, weights],
        ['softmax', 'loss'],
    )

    if not args.forward_only:
        model.AddGradientOperators([loss])

    # carry states over
    for init_blob in init_blobs:
        model.net.Copy(last_hidden, init_blob)

        sz = args.hidden_dim
        if args.implementation == "cudnn":
            sz *= args.num_layers
        workspace.FeedBlob(init_blob, np.zeros(
            [1, args.batch_size, sz], dtype=np.float32
        ))

    if args.rnn_executor:
        for op in model.net.Proto().op:
            if op.type.startswith('RecurrentNetwork'):
                recurrent.set_rnn_executor_config(
                    op,
                    num_threads=args.rnn_executor_num_threads,
                    max_cuda_streams=args.rnn_executor_max_cuda_streams,
                )
    return model, output


def Caffe2LSTM(args):
    T = args.data_size // args.batch_size

    input_blob_shape = [args.seq_length, args.batch_size, args.input_dim]
    queue, label_queue, entry_counts = generate_data(T // args.seq_length,
                                       input_blob_shape,
                                       args.hidden_dim,
                                       args.fixed_shape)

    workspace.FeedBlob(
        "seq_lengths",
        np.array([args.seq_length] * args.batch_size, dtype=np.int32)
    )

    model, output = create_model(args, queue, label_queue, input_blob_shape)

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)

    start_time = time.time()
    num_iters = T // args.seq_length
    total_iters = 0

    # Run the Benchmark
    log.info("------ Warming up ------")
    workspace.RunNet(model.net.Proto().name)

    if (args.gpu):
        log.info("Memory stats:")
        stats = utils.GetGPUMemoryUsageStats()
        log.info("GPU memory:\t{} MB".format(stats['max_total'] / 1024 / 1024))

    log.info("------ Starting benchmark ------")
    start_time = time.time()
    last_time = time.time()
    for iteration in range(1, num_iters, args.iters_to_report):
        iters_once = min(args.iters_to_report, num_iters - iteration)
        total_iters += iters_once
        workspace.RunNet(model.net.Proto().name, iters_once)

        new_time = time.time()
        log.info(
            "Iter: {} / {}. Entries Per Second: {}k.".format(
                iteration,
                num_iters,
                np.sum(entry_counts[iteration:iteration + iters_once]) /
                (new_time - last_time) // 100 / 10,
            )
        )
        last_time = new_time

    log.info("Done. Total EPS excluding 1st iteration: {}k {}".format(
         np.sum(entry_counts[1:]) / (time.time() - start_time) // 100 / 10,
         " (with RNN executor)" if args.rnn_executor else "",
    ))

    if (args.gpu):
        log.info("Memory stats:")
        stats = utils.GetGPUMemoryUsageStats()
        log.info("GPU memory:\t{} MB".format(stats['max_total'] / 1024 / 1024))
        if (stats['max_total'] != stats['total']):
            log.warning(
                "Max usage differs from current total usage: {} > {}".
                format(stats['max_total'], stats['total'])
            )
            log.warning("This means that costly deallocations occurred.")

    return time.time() - start_time


@utils.debug
def Benchmark(args):
    return Caffe2LSTM(args)


def GetArgumentParser():
    parser = argparse.ArgumentParser(description="LSTM benchmark.")

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=800,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=40,
        help="Input dimension",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The batch size."
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=20,
        help="Max sequence length"
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=1000000,
        help="Number of data points to generate"
    )
    parser.add_argument(
        "--iters_to_report",
        type=int,
        default=20,
        help="Number of iteration to report progress"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run all on GPU",
    )
    parser.add_argument(
        "--implementation",
        type=str,
        default="own",
        help="'cudnn', 'own', 'static' or 'static_dag'",
    )
    parser.add_argument(
        "--fixed_shape",
        action="store_true",
        help=("Whether to randomize shape of input batches. "
              "Static RNN requires fixed shape"),
    )
    parser.add_argument(
        "--memory_optimization",
        action="store_true",
        help="Whether to use memory optimized LSTM or not",
    )
    parser.add_argument(
        "--forward_only",
        action="store_true",
        help="Whether to run only forward pass"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of LSTM layers. All output dimensions are going to be"
             "of hidden_dim size",
    )
    parser.add_argument(
        "--rnn_executor",
        action="store_true",
        help="Whether to use RNN executor"
    )
    parser.add_argument(
        "--rnn_executor_num_threads",
        type=int,
        default=None,
        help="Number of threads used by CPU RNN Executor"
    )
    parser.add_argument(
        "--rnn_executor_max_cuda_streams",
        type=int,
        default=None,
        help="Maximum number of CUDA streams used by RNN executor on GPU"
    )
    return parser


if __name__ == '__main__':
    args, extra_args = GetArgumentParser().parse_known_args()

    rnn_executor_opt = 1 if args.rnn_executor else 0

    workspace.GlobalInit([
        'caffe2',
        '--caffe2_log_level=0',
        '--caffe2_print_blob_sizes_at_exit=0',
        '--caffe2_rnn_executor={}'.format(rnn_executor_opt),
        '--caffe2_gpu_memory_tracking=1'] + extra_args)

    device = core.DeviceOption(
        workspace.GpuDeviceType if args.gpu else caffe2_pb2.CPU, 4)

    with core.DeviceScope(device):
        Benchmark(args)
