import numpy as np

# Construction

np.float32(3j)  # E: incompatible type

# Technically the following examples are valid NumPy code. But they
# are not considered a best practice, and people who wish to use the
# stubs should instead do
#
# np.array([1.0, 0.0, 0.0], dtype=np.float32)
# np.array([], dtype=np.complex64)
#
# See e.g. the discussion on the mailing list
#
# https://mail.python.org/pipermail/numpy-discussion/2020-April/080566.html
#
# and the issue
#
# https://github.com/numpy/numpy-stubs/issues/41
#
# for more context.
np.float32([1.0, 0.0, 0.0])  # E: incompatible type
np.complex64([])  # E: incompatible type

np.complex64(1, 2)  # E: Too many arguments
# TODO: protocols (can't check for non-existent protocols w/ __getattr__)

np.datetime64(0)  # E: non-matching overload

class A:
    def __float__(self):
        return 1.0


np.int8(A())  # E: incompatible type
np.int16(A())  # E: incompatible type
np.int32(A())  # E: incompatible type
np.int64(A())  # E: incompatible type
np.uint8(A())  # E: incompatible type
np.uint16(A())  # E: incompatible type
np.uint32(A())  # E: incompatible type
np.uint64(A())  # E: incompatible type

np.void("test")  # E: incompatible type

np.generic(1)  # E: Cannot instantiate abstract class
np.number(1)  # E: Cannot instantiate abstract class
np.integer(1)  # E: Cannot instantiate abstract class
np.inexact(1)  # E: Cannot instantiate abstract class
np.character("test")  # E: Cannot instantiate abstract class
np.flexible(b"test")  # E: Cannot instantiate abstract class

np.float64(value=0.0)  # E: Unexpected keyword argument
np.int64(value=0)  # E: Unexpected keyword argument
np.uint64(value=0)  # E: Unexpected keyword argument
np.complex128(value=0.0j)  # E: Unexpected keyword argument
np.str_(value='bob')  # E: No overload variant
np.bytes_(value=b'test')  # E: No overload variant
np.void(value=b'test')  # E: Unexpected keyword argument
np.bool_(value=True)  # E: Unexpected keyword argument
np.datetime64(value="2019")  # E: No overload variant
np.timedelta64(value=0)  # E: Unexpected keyword argument

np.bytes_(b"hello", encoding='utf-8')  # E: No overload variant
np.str_("hello", encoding='utf-8')  # E: No overload variant

complex(np.bytes_("1"))  # E: No overload variant
