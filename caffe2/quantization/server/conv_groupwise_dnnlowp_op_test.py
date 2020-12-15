

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
from caffe2.python import core, dyndep, workspace
from caffe2.python.fb import hardcode_scale_zp  # type: ignore[import]
from caffe2.quantization.server import utils as dnnlowp_utils
from caffe2.quantization.server.dnnlowp_test_utils import (
    check_quantized_results_close,
    generate_conv_inputs,
    run_conv_or_fc,
)
from hypothesis import assume, given, settings


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class GroupWiseDNNLowPOpConvTest(hu.HypothesisTestCase):
    # correctness test with no quantization error in inputs
    @given(
        stride=st.integers(1, 2),
        pad=st.integers(0, 2),
        kernel=st.integers(1, 5),
        dilation=st.integers(1, 2),
        size=st.integers(10, 16),
        group=st.integers(1, 4),
        input_channels_per_group=st.sampled_from([2, 3, 4, 5, 8, 16, 32]),
        output_channels_per_group=st.integers(2, 16),
        batch_size=st.integers(0, 3),
        order=st.sampled_from(["NCHW", "NHWC"]),
        prepack_weight=st.booleans(),
        preserve_activation_sparsity=st.booleans(),
        preserve_weight_sparsity=st.booleans(),
        **hu.gcs_cpu_only
    )
    @settings(max_examples=10, deadline=None)
    def test_groupwise_dnnlowp_conv_int(
        self,
        stride,
        pad,
        kernel,
        dilation,
        size,
        group,
        input_channels_per_group,
        output_channels_per_group,
        batch_size,
        order,
        prepack_weight,
        preserve_activation_sparsity,
        preserve_weight_sparsity,
        gc,
        dc,
    ):
        assume(group == 1 or dilation == 1)
        assume((not prepack_weight) or order == "NHWC")

        X, W, b = generate_conv_inputs(
            stride,
            pad,
            kernel,
            dilation,
            size,
            group,
            input_channels_per_group,
            output_channels_per_group,
            batch_size,
            order,
            groupwise_quantization=True,
            preserve_activation_sparsity=preserve_activation_sparsity,
            preserve_weight_sparsity=preserve_weight_sparsity,
        )

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
        outputs = []

        op_engine_list = [
            ("Conv", ""),
            ("Conv", "DNNLOWP"),
            ("Conv", "DNNLOWP_16"),
            ("Int8Conv", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            init_net = core.Net("test_init_net")
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine
            do_dequantize = "DNNLOWP" in engine
            do_prepack_weight = engine == "DNNLOWP" and prepack_weight

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize",
                    ["X"],
                    ["X_q"],
                    preserve_activation_sparsity=preserve_activation_sparsity,
                    engine=engine,
                    device_option=gc,
                )
                net.Proto().op.extend([quantize])

            if do_prepack_weight:
                X_min = 0 if X.size == 0 else X.min()
                X_max = 0 if X.size == 0 else X.max()
                x_q_param = hardcode_scale_zp.choose_quantization_params(X_min, X_max)
                inputs = ["W"]
                if do_dequantize:
                    inputs += ["b"]
                pack = core.CreateOperator(
                    "Int8ConvPackWeight",
                    inputs,
                    ["W_packed"],
                    stride=stride,
                    kernel=kernel,
                    dilation=dilation,
                    pad=pad,
                    preserve_weight_sparsity=preserve_weight_sparsity,
                    engine=engine,
                    group=group,
                    quantize_groupwise=1,
                    in_scale=x_q_param.scale,
                )
                init_net.Proto().op.extend([pack])

            conv = core.CreateOperator(
                op_type,
                [
                    "X_q" if do_quantize else "X",
                    "W_packed" if do_prepack_weight else "W",
                    "b",
                ],
                ["Y_q" if do_dequantize else "Y"],
                stride=stride,
                kernel=kernel,
                dilation=dilation,
                pad=pad,
                order=order,
                preserve_activation_sparsity=preserve_activation_sparsity,
                preserve_weight_sparsity=preserve_weight_sparsity,
                engine=engine,
                group=group,
                quantize_groupwise=1,
                device_option=gc,
            )
            if do_dequantize or do_prepack_weight:
                # groupwise quantization only works with static quantization
                # so we need to set quantization parameters
                dnnlowp_utils.add_quantization_param_args(
                    conv, outputs[0][0], preserve_activation_sparsity
                )
            net.Proto().op.extend([conv])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize",
                    ["Y_q"],
                    ["Y"],
                    preserve_activation_sparsity=preserve_activation_sparsity,
                    engine=engine,
                    device_option=gc,
                )
                net.Proto().op.extend([dequantize])

            run_conv_or_fc(
                self, init_net, net, X, W, b, op_type, engine, order, gc, outputs
            )

        check_quantized_results_close(outputs, symmetric=preserve_activation_sparsity)

    # correctness test with no quantization error in inputs
    @given(
        stride=st.integers(1, 2),
        pad=st.integers(0, 2),
        kernel=st.integers(1, 5),
        dilation=st.integers(1, 2),
        size=st.integers(10, 16),
        group=st.integers(1, 4),
        input_channels_per_group=st.sampled_from([2, 3, 4, 5, 8, 16, 32]),
        output_channels_per_group=st.integers(2, 16),
        batch_size=st.integers(0, 3),
        order=st.sampled_from(["NCHW", "NHWC"]),
        **hu.gcs_cpu_only
    )
    @settings(max_examples=10, deadline=None)
    def test_groupwise_dnnlowp_conv_relu_int(
        self,
        stride,
        pad,
        kernel,
        dilation,
        size,
        group,
        input_channels_per_group,
        output_channels_per_group,
        batch_size,
        order,
        gc,
        dc,
    ):
        assume(group == 1 or dilation == 1)

        X, W, b = generate_conv_inputs(
            stride,
            pad,
            kernel,
            dilation,
            size,
            group,
            input_channels_per_group,
            output_channels_per_group,
            batch_size,
            order,
            True,  # group-wise
        )

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
        outputs = []

        op_engine_list = [
            ("Conv", ""),
            ("ConvRelu", "DNNLOWP"),
            ("ConvRelu", "DNNLOWP_16"),
            ("Int8ConvRelu", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            if "DNNLOWP" in engine:
                quantize = core.CreateOperator(
                    "Quantize", ["X"], ["X_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize])

                conv = core.CreateOperator(
                    op_type,
                    ["X_q", "W", "b"],
                    ["Y_q"],
                    stride=stride,
                    kernel=kernel,
                    dilation=dilation,
                    pad=pad,
                    order=order,
                    engine=engine,
                    group=group,
                    quantize_groupwise=1,
                    device_option=gc,
                )
                # groupwise quantization only works with static quantization
                # so we need to set quantization parameters
                dnnlowp_utils.add_quantization_param_args(conv, outputs[0][0])
                net.Proto().op.extend([conv])

                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])
            else:
                conv = core.CreateOperator(
                    op_type,
                    ["X", "W", "b"],
                    ["Y"],
                    stride=stride,
                    kernel=kernel,
                    dilation=dilation,
                    pad=pad,
                    order=order,
                    engine=engine,
                    group=group,
                    device_option=gc,
                )
                net.Proto().op.extend([conv])

                relu = core.CreateOperator(
                    "Relu", ["Y"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([relu])

            run_conv_or_fc(
                self, None, net, X, W, b, op_type, engine, order, gc, outputs
            )

        check_quantized_results_close(outputs)
