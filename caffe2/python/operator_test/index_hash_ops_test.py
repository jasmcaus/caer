




from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np


class TestIndexHashOps(serial.SerializedTestCase):
    @given(
        indices=st.sampled_from([
            np.int32, np.int64
        ]).flatmap(lambda dtype: hu.tensor(min_dim=1, max_dim=1, dtype=dtype)),
        seed=st.integers(min_value=0, max_value=10),
        modulo=st.integers(min_value=100000, max_value=200000),
        **hu.gcs_cpu_only
    )
    @settings(deadline=10000)
    def test_index_hash_ops(self, indices, seed, modulo, gc, dc):
        def index_hash(indices):
            dtype = np.array(indices).dtype
            assert dtype == np.int32 or dtype == np.int64
            hashed_indices = []
            for index in indices:
                hashed = dtype.type(0xDEADBEEF * seed)
                indices_bytes = np.array([index], dtype).view(np.int8)
                for b in indices_bytes:
                    hashed = dtype.type(hashed * 65537 + b)
                hashed = (modulo + hashed % modulo) % modulo
                hashed_indices.append(hashed)
            return [hashed_indices]

        op = core.CreateOperator("IndexHash",
                                 ["indices"], ["hashed_indices"],
                                 seed=seed, modulo=modulo)

        self.assertDeviceChecks(dc, op, [indices], [0])
        self.assertReferenceChecks(gc, op, [indices], index_hash)

        # In-place update
        op = core.CreateOperator("IndexHash",
                                 ["indices"], ["indices"],
                                 seed=seed, modulo=modulo)

        self.assertDeviceChecks(dc, op, [indices], [0])
        self.assertReferenceChecks(gc, op, [indices], index_hash)

    def test_shape_and_type_inference(self):
        with hu.temp_workspace("shape_type_inf_int64"):
            net = core.Net('test_net')
            net.ConstantFill(
                [], "values", shape=[64], dtype=core.DataType.INT64,
            )
            net.IndexHash(['values'], ['values_output'])
            (shapes, types) = workspace.InferShapesAndTypes([net], {})

            self.assertEqual(shapes["values_output"], [64])
            self.assertEqual(types["values_output"], core.DataType.INT64)

        with hu.temp_workspace("shape_type_inf_int32"):
            net = core.Net('test_net')
            net.ConstantFill(
                [], "values", shape=[2, 32], dtype=core.DataType.INT32,
            )
            net.IndexHash(['values'], ['values_output'])
            (shapes, types) = workspace.InferShapesAndTypes([net], {})

            self.assertEqual(shapes["values_output"], [2, 32])
            self.assertEqual(types["values_output"], core.DataType.INT32)
