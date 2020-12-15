




from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


class TestUnmaskOp(serial.SerializedTestCase):
    @serial.given(N=st.integers(min_value=2, max_value=20),
           dtype=st.sampled_from([
               np.bool_,
               np.int8,
               np.int16,
               np.int32,
               np.int64,
               np.uint8,
               np.uint16,
               np.float16,
               np.float32,
               np.float64]),
           **hu.gcs)
    def test(self, N, dtype, gc, dc):
        if dtype is np.bool_:
            all_value = np.random.choice(a=[True, False], size=N)
        else:
            all_value = (np.random.rand(N) * N).astype(dtype)

        M = np.random.randint(1, N)
        split = sorted(np.random.randint(1, N, size=M))
        indices = np.random.permutation(N)
        pieces = np.split(indices, split)

        def ref(*args, **kwargs):
            return (all_value,)

        inputs = []
        inputs_names = []
        for i, piece in enumerate(pieces):
            piece.sort()
            mask = np.zeros(N, dtype=np.bool_)
            mask[piece] = True
            values = all_value[piece]
            inputs.extend([mask, values])
            inputs_names.extend(["mask%d" % i, "value%d" % i])

        op = core.CreateOperator(
            'BooleanUnmask',
            inputs_names,
            'output')

        self.assertReferenceChecks(gc, op, inputs, ref)
        self.assertDeviceChecks(dc, op, inputs, [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
