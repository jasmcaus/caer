## @package sparse_lookup
# Module caffe2.python.layers.sparse_lookup





from caffe2.python.optimizer import FP16_ENGINES, Optimizer
from caffe2.python.helpers.arg_scope import get_current_scope
from caffe2.python import schema
from caffe2.python.layers.layers import (
    get_categorical_limit,
    get_key,
    IdList,
    IdScoreList,
    IdListWithEvicted,
    IdScoreListWithEvicted,
    LayerPsParam,
    ModelLayer,
    almost_equal_schemas,
)
import collections
import functools
import logging
import math
import numpy as np
import operator

logger = logging.getLogger(__name__)


def get_trainer_version_based_on_optim(optim_def):
    if isinstance(optim_def, Optimizer) and hasattr(optim_def, "engine"):
        logger.info(
            "Attempting to set trainer version for engine {}".format(optim_def.engine)
        )
        if optim_def.engine in FP16_ENGINES:
            logger.info("Setting FP16 trainer for engine {}".format(optim_def.engine))
            return "fp16"
        else:
            logger.info("Setting FP32 trainer for engine {}".format(optim_def.engine))
            return "fp32"
    else:
        return "fp32"


def get_sparse_lookup_predictor_version(
    version,
    blob_size=None,
    min_blob_size_4bits=None,
    embedding_dim=None,
    sparse_feature_name=None,
):
    assert version in {
        'fp32', 'fp16', 'uint8rowwise', 'fused_uint8rowwise', 'fused_uint4rowwise'
    }, "Unexpected version of sparse_lookup layer {0}".format(version)
    if version == 'fused_uint4rowwise':
        if (
            blob_size is not None
            and min_blob_size_4bits is not None
            and embedding_dim is not None
        ):
            if blob_size < min_blob_size_4bits:
                logger.info(
                    "{} fall back to uint8 because lookup table size {} < min_blob_size_4bits {}".format(
                        sparse_feature_name,
                        blob_size,
                        min_blob_size_4bits,
                    )
                )
                version = 'fused_uint8rowwise'

            if embedding_dim % 2 == 1:
                logger.info(
                    "{} fall back to uint8 because lookup table dimension {} is not divisible by 2".format(
                        sparse_feature_name, embedding_dim
                    )
                )
                version = 'fused_uint8rowwise'
        else:
            raise ValueError(
                (
                    "When 4 bit quantization is enabled for {}, "
                    "(i.e., Sparse lookup predictor version:{}), "
                    "requires arguments blob_size:{}, "
                    "min_blob_size_4bits:{}, embedding_dim:{}"
                ).format(
                    sparse_feature_name,
                    version,
                    blob_size,
                    min_blob_size_4bits,
                    embedding_dim
                )
            )
    return version


def get_sparse_lookup_trainer_version(version):
    assert version in {'fp32', 'fp16'},\
        "Unexpected version of sparse_lookup layer {0}".format(version)
    return version

def _is_id_list(input_record):
    return almost_equal_schemas(input_record, IdList)


def _is_id_score_list(input_record):
    return almost_equal_schemas(input_record,
                                IdScoreList,
                                check_field_types=False)


class SparseLookup(ModelLayer):
    _id_list_supported_reducers = [
        'LogMeanExp', 'LogSumExp', 'Max', 'Mean', 'Sum',
        'WeightedSum', 'WeightedMean', 'Sqrt', 'None']

    _id_score_list_supported_reducers = [
        'PositionWeighted', 'RecencyWeighted', 'Mean', 'Sum', 'WeightedSum',
        'WeightedMean', 'None'
    ]

    _fp16_compatible_init_op_types = [
        'Float16UniformFill'
    ]

    _fp16_compatible_reducers = [
        'Sum', 'Mean', 'Sqrt', 'PositionWeighted', 'RecencyWeighted',
    ]

    def __init__(self, model, input_record, inner_shape, reducer,
                 weight_init=None, weight_optim=None,
                 name='sparse_lookup', regularizer=None, use_external_weights=False,
                 uniform_weight_init_scale_numerator=1.0, **kwargs):

        super(SparseLookup, self).__init__(model, name, input_record, **kwargs)

        self.sparse_key = get_key(self.input_record)()
        logger.info("Setup the sparse lookup layer for " + self.sparse_key)

        # TODO Add some asserts about input type
        if isinstance(inner_shape, int):
            inner_shape = [inner_shape]
        assert isinstance(inner_shape, list) or isinstance(inner_shape, tuple),\
            "Unexpected type for inner_shape, expected list or tuple, got {0} for {1}".\
            format(type(inner_shape), self.sparse_key)

        if reducer == "PositionWeighted":
            assert _is_id_score_list(self.input_record), (
                "PositionWeighted only support IdScoreList, but got {} for {}"
                + "please use PositionWeighted layer to convert IdList "
                + "to IdScoreList"
            ).format(repr(self.input_record), self.sparse_key)
            self.external_weights = self.input_record.values()

        elif reducer == "RecencyWeighted":
            assert _is_id_score_list(self.input_record), (
                "RecencyWeighted only supports IdScoreList, "
                "while the sparse feature {} is not.".format(self.sparse_key)
            )
            self.external_weights = self.input_record.values()
        # TODO: create a new type of reducer with external weights to wrap
        # this and the above two cases since essentially their input formats
        # are the same.
        elif use_external_weights:
            assert _is_id_score_list(self.input_record), (
                "Use_external_weights only supports IdScoreList, "
                "while the sparse feature {} is not.".format(self.sparse_key)
            )
            assert reducer in ["Sum", "WeightedSum"], (
                "Use_external_weights only supports Sum reducer, "
                "while the reducer is {}.".format(reducer)
            )
            self.external_weights = self.input_record.values()
        self.reducer = reducer
        self.use_external_weights = use_external_weights

        input_dim = get_categorical_limit(self.input_record)
        assert input_dim > 0, "{} should have categorical limit > 0, but got {}".format(
            self.sparse_key, input_dim
        )

        self.input_dim = input_dim
        self.shape = [input_dim] + inner_shape

        self.trainer_version = get_trainer_version_based_on_optim(
            weight_optim
        )

        self.uniform_weight_init_scale_numerator = uniform_weight_init_scale_numerator
        default_init_op = self._get_default_init_op()

        self.weight_init = weight_init or default_init_op

        self.evicted_values = None
        if schema.equal_schemas(
            self.input_record, IdListWithEvicted
        ) or schema.equal_schemas(
            self.input_record, IdScoreListWithEvicted, check_field_types=False
        ):
            self.evicted_values = self.input_record._evicted_values

        # If fp16 is used, make sure fp16 init op is used
        if self.trainer_version == "fp16":
            assert self.reducer in self._fp16_compatible_reducers or use_external_weights, (
                "Fp16 training is enabled. The reducer specified is not supported. "
                "Got {}. Supported reducers: {}. Right now, in general, sum, mean, "
                "positional pooling are supported. Attention is not. Please check "
                "if there is fp16 trained sparse features using advanced pooling.".format(
                    self.reducer, self._fp16_compatible_reducers)
            )

            # if init op is UniformFill, we replace it directly
            if self.weight_init[0] == "UniformFill":
                self.weight_init = ("Float16UniformFill", self.weight_init[1])
            assert self.weight_init[0] in self._fp16_compatible_init_op_types, (
                "Fp16 training is enabled. Init op for weight parameter must be fp16 "
                "compatibale. Got {}. Supported ops: {}".format(
                    self.weight_init[0],
                    self._fp16_compatible_init_op_types)
            )

            assert regularizer is None, "Regularizer is not compatible with fp16"

        if self.input_record.lengths.metadata:
            avg_length = self.input_record.lengths.metadata.expected_value
        else:
            avg_length = None

        self.w = self.create_param(
            param_name='w',
            shape=self.shape,
            initializer=self.weight_init,
            optimizer=weight_optim,
            ps_param=LayerPsParam(
                sparse_key=self.sparse_key,
                average_length=avg_length),
            regularizer=regularizer
        )
        if self.evicted_values:
            self.reinit_vec = self.create_param(
                param_name="reinit_vec",
                shape=inner_shape,
                initializer=self.weight_init,
                optimizer=model.NoOptim,
                regularizer=None,
            )

        self.scale_bias_init = ('ConstantFill', {'value': 0.0})

        self.scale_bias = self.create_param(
            param_name='scale_bias',
            shape=[],
            initializer=self.scale_bias_init,
            optimizer=model.NoOptim,
        )

        self.output_schema = schema.Scalar(
            (np.float32, inner_shape),
            self.get_next_blob_reference('output'),
        )

    def get_memory_usage(self):
        return functools.reduce(operator.mul, self.shape) * 4

    def get_fp16_compatible_parameters(self):
        return [self.w]

    def support_8bit(self):
        # Rowwise quantization makes sense only if shape it's 2D matrix with
        # second dimension >= 8
        if len(self.shape) != 2 or self.shape[1] < 8:
            return False
        return True

    def get_8bits_compatible_parameters(self, fused=True):
        if not self.support_8bit():
            return []
        if fused:
            RowwiseQuantized8BitsWeight = collections.namedtuple(
                'RowwiseQuantized8BitsWeight', 'w'
            )
            return [RowwiseQuantized8BitsWeight(self.w)]
        else:
            RowwiseQuantized8BitsWeight = collections.namedtuple(
                'RowwiseQuantized8BitsWeight', 'w, scale_bias'
            )
            return [RowwiseQuantized8BitsWeight(self.w, self.scale_bias)]

    def _get_default_init_op(self):
        scale = math.sqrt(self.uniform_weight_init_scale_numerator / self.input_dim)

        if self.trainer_version == 'fp32':
            default_weight_init = ('UniformFill', {'min': -scale, 'max': scale})
        elif self.trainer_version == 'fp16':
            default_weight_init = ("Float16UniformFill", {'min': -scale, 'max': scale})
        else:
            raise NotImplementedError(
                "Train version {} is not currently supported for sparse feature {}".format(
                    trainer_version, self.sparse_key
                )
            )

        return default_weight_init

    def _gather_wrapper(self, net, version, in_indices, out):
        # Gather can work on all kinds of input data types, and output
        # data with the same type. Convert the output of Gather to float,
        # because the follow-up Ops expect fp32.
        if version == 'fp32':
            return net.Gather([self.w, in_indices], out)
        elif version == 'fp16':
            gathered_w = net.Gather([self.w, in_indices], 'gathered_w')
            return net.HalfToFloat(gathered_w, out)
        elif version == 'uint8rowwise':
            gathered_w = net.Gather([self.w, in_indices], 'gathered_w')
            gathered_scale_bias = net.Gather(
                [self.scale_bias, in_indices],
                'gathered_scale_bias'
            )

            return net.Rowwise8BitQuantizedToFloat(
                [gathered_w, gathered_scale_bias], out)
        elif version == 'fused_uint8rowwise':
            gathered_w = net.Gather([self.w, in_indices], 'gathered_w')
            return net.Fused8BitRowwiseQuantizedToFloat(gathered_w, out)
        elif version == 'fused_uint4rowwise':
            gathered_w = net.Gather([self.w, in_indices], 'gathered_w')
            return net.Fused4BitRowwiseQuantizedToFloat(gathered_w, out)

        else:
            raise "Unsupported version of operators in SparseLookup " +\
                "layer: {0} for sparse feature {1}".format(
                    version, self.sparse_key
                )

    def _sparse_lengths_weighted_reducer(
        self,
        in_indices,
        weights,
        reducer,
        net,
        version,
        grad_on_weights=0,
    ):
        op_input = [
            self.w,
            weights,
            in_indices,
            self.input_record.lengths(),
        ]
        layer_name = 'SparseLengths' + reducer

        if version in ['fp32', 'fp16']:
            # SparseLengths* Ops will accept either fp16 or fp32 embedding
            # matrix and output fp32 pooled embedding
            # A special case here is that we need FP16 engine for
            # SparseLengthsWeightedSum when FP16 embeedings are used for
            # correct backward updates
            if reducer == "WeightedSum" and version == "fp16":
                net.SparseLengthsWeightedSum(
                    op_input,
                    self.output_schema.field_blobs(),
                    grad_on_weights=grad_on_weights,
                    engine='FP16',
                )
            else:
                net.__getattr__(layer_name)(
                    op_input,
                    self.output_schema.field_blobs(),
                    grad_on_weights=grad_on_weights,
                )
        elif version == 'uint8rowwise':
            op_input.insert(len(op_input), self.scale_bias)
            net.__getattr__(layer_name + '8BitsRowwise')(
                op_input, self.output_schema.field_blobs())
        elif version == 'fused_uint8rowwise':
            net.__getattr__(layer_name + 'Fused8BitRowwise')(
                op_input, self.output_schema.field_blobs())
        elif version == 'fused_uint4rowwise':
            net.__getattr__(layer_name + 'Fused4BitRowwise')(
                op_input, self.output_schema.field_blobs())
        else:
            raise "Unsupported version of operator in SparseLookUp " +\
                "layer: {0} for sparse feature {1}".format(
                    version, self.sparse_key
                )

    # deal with sparse features of id_list type
    def _add_ops_id_list(self, net, version):
        assert self.reducer in self._id_list_supported_reducers, (
            "Unsupported reducer: {} for ID_LIST {}".format(
                self.reducer, self.sparse_key
            )
        )
        if self.reducer in ['Sum', 'Mean', 'WeightedSum', 'WeightedMean']:
            op_input = [self.w,
                        self.input_record.items(),
                        self.input_record.lengths()]

            # For id list features, the behaviors of 'Sum' and
            # 'WeightedSum' are identical, since we can regard the weight on each
            # id as 1. Similarly, for 'Mean' and 'WeightedMean'.
            if self.reducer == 'WeightedSum':
                self.reducer = 'Sum'
            elif self.reducer == 'WeightedMean':
                self.reducer = 'Mean'

            layer_name = 'SparseLengths' + self.reducer
            if version in ['fp32', 'fp16']:
                # SparseLengths* Ops will accept either fp16 or fp32 embedding
                # matrix and output fp32 pooled embedding
                net.__getattr__(layer_name)(
                    op_input,
                    self.output_schema.field_blobs(),
                )
            elif version == 'uint8rowwise':
                op_input.insert(len(op_input), self.scale_bias)
                net.__getattr__(layer_name + '8BitsRowwise')(
                    op_input, self.output_schema.field_blobs())
            elif version == 'fused_uint8rowwise':
                net.__getattr__(layer_name + 'Fused8BitRowwise')(
                    op_input, self.output_schema.field_blobs())
            elif version == 'fused_uint4rowwise':
                net.__getattr__(layer_name + 'Fused4BitRowwise')(
                    op_input, self.output_schema.field_blobs())
            else:
                raise "Unsupported version of operator in SparseLookUp " +\
                    "layer: {0} for sparse feature {1}".format(
                        version, self.sparse_key
                    )

        elif self.reducer == 'Sqrt':
            sqrt_weight = net.LengthsToWeights(
                [self.input_record.lengths()],
                [net.NextScopedBlob('lengths_sqrt')],
                power=0.5,
            )
            self._sparse_lengths_weighted_reducer(
                self.input_record.items(),
                sqrt_weight,
                'WeightedSum', net, version)

        elif self.reducer == 'None':
            # Gather operator will gather the embedding for each id of
            # each IdList.
            self._gather_wrapper(net, version, self.input_record.items(),
                                 self.output_schema.field_blobs())

        else:
            table_rows = self._gather_wrapper(
                net, version, self.input_record.items(), 'table_rows')

            segment_ids = net.LengthsToSegmentIds(
                self.input_record.lengths(),
                net.NextScopedBlob(self.input_record.lengths() + '_sid'))
            net.__getattr__('SortedSegmentRange' + self.reducer)(
                [table_rows, segment_ids],
                self.output_schema.field_blobs(),
            )

    # deal with sparse features of id_score_list type
    def _add_ops_id_score_list(self, net, version):
        assert self.reducer in self._id_score_list_supported_reducers, (
            "Unsupported reducer: {} for ID_SCORE_LIST {}".format(
                self.reducer, self.sparse_key
            )
        )
        if self.reducer in ['WeightedSum', 'WeightedMean']:
            self._sparse_lengths_weighted_reducer(
                self.input_record.keys(),
                self.input_record.values(),
                self.reducer, net, version)

        elif self.reducer in ['PositionWeighted', 'RecencyWeighted'] or self.use_external_weights:
            self._sparse_lengths_weighted_reducer(
                self.input_record.keys(),
                self.external_weights,
                'WeightedSum', net, version, grad_on_weights=1)

        elif self.reducer in ['Sum', 'Mean']:
            op_input = [self.w,
                        self.input_record.keys(),
                        self.input_record.lengths()]

            layer_name = 'SparseLengths' + self.reducer

            if version in ['fp32', 'fp16']:
                net.__getattr__(layer_name)(
                    op_input,
                    self.output_schema.field_blobs(),
                )
            elif version == 'uint8rowwise':
                net.__getattr__(layer_name + '8BitsRowwise')(
                    op_input, self.output_schema.field_blobs())
            elif version == 'fused_uint8rowwise':
                net.__getattr__(layer_name + 'Fused8BitRowwise')(
                    op_input, self.output_schema.field_blobs())
            elif version == 'fused_uint4rowwise':
                net.__getattr__(layer_name + 'Fused4BitRowwise')(
                    op_input, self.output_schema.field_blobs())
            else:
                raise "Unsupported version of operator in SparseLookUp " +\
                    "layer: {0} for sparse feature {1}".format(
                        version, self.sparse_key
                    )

        elif self.reducer == 'None':
            # Gather operator will gather the embedding for each id of
            # each IdList.
            self._gather_wrapper(net, version, self.input_record.keys(),
                                 self.output_schema.field_blobs())
        else:
            raise "Only Sum, Mean, None are supported for IdScoreList input." +\
                "Trying to create with {} for sparse feature {}".format(
                    self.reducer, self.sparse_key
                )

    def _add_ops(self, net, version='fp32', is_train=True):
        if self.evicted_values and is_train:
            net.CopyRowsToTensor(
                [self.w, self.evicted_values.get(), self.reinit_vec], [self.w])
        if _is_id_list(self.input_record):
            self._add_ops_id_list(net, version=version)
        elif _is_id_score_list(self.input_record):
            self._add_ops_id_score_list(net, version=version)
        else:
            raise "Unsupported input type {0}".format(self.input_record)

    def add_train_ops(self, net):
        self._add_ops(net, self.trainer_version, is_train=True)

    def add_ops(self, net):
        version_info = get_current_scope().get(
            get_sparse_lookup_predictor_version.__name__, {'version': 'fp32'}
        )
        lookup_table_blob_size = self.shape[0] * self.shape[1]
        version = get_sparse_lookup_predictor_version(
            version_info['version'],
            blob_size=lookup_table_blob_size,
            min_blob_size_4bits=(
                version_info['min_blob_size_4bits']
                if 'min_blob_size_4bits' in version_info
                else None
            ),
            embedding_dim=self.shape[1],
            sparse_feature_name=self.sparse_key,
        )

        # TODO(amalevich): Layer should not be responsible for decision about
        # quantization.
        if not self.support_8bit() and version in {'uint8rowwise',
                                                   'fused_uint8rowwise',
                                                   'fused_uint4rowwise'}:
            version = 'fp16'

        self._add_ops(net, version, is_train=False)
