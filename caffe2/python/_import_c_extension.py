## @package _import_c_extension
# Module caffe2.python._import_c_extension
import atexit
import logging
import sys
from caffe2.python import extension_loader

# NOTE: we have to import python protobuf here **before** we load cpp extension.
# Otherwise it breaks under certain build conditions if cpp implementation of
# protobuf is used. Presumably there's some registry in protobuf library and
# python side has to initialize the dictionary first, before static
# initialization in python extension does so. Otherwise, duplicated protobuf
# descriptors will be created and it can lead to obscure errors like
#   "Parameter to MergeFrom() must be instance of same class:
#    expected caffe2.NetDef got caffe2.NetDef."
import caffe2.proto

# We will first try to load the gpu-enabled caffe2. If it fails, we will then
# attempt to load the cpu version. The cpu backend is the minimum required, so
# if that still fails, we will exit loud.
with extension_loader.DlopenGuard():
    has_hip_support = False
    has_cuda_support = False
    has_gpu_support = False

    try:
        from caffe2.python.caffe2_pybind11_state_gpu import *  # noqa
        if num_cuda_devices():  # noqa
            has_gpu_support = has_cuda_support = True
    except ImportError as gpu_e:
        logging.info('Failed to import cuda module: {}'.format(gpu_e))
        try:
            from caffe2.python.caffe2_pybind11_state_hip import *  # noqa
            # we stop checking whether we have AMD GPU devices on the host,
            # because we may be constructing a net on a machine without GPU,
            # and run the net on another one with GPU
            has_gpu_support = has_hip_support = True
            logging.info('This caffe2 python run has AMD GPU support!')
        except ImportError as hip_e:
            logging.info('Failed to import AMD hip module: {}'.format(hip_e))

            logging.warning(
                'This caffe2 python run failed to load cuda module:{},'
                'and AMD hip module:{}.'
                'Will run in CPU only mode.'.format(gpu_e, hip_e))
            try:
                from caffe2.python.caffe2_pybind11_state import *  # noqa
            except ImportError as cpu_e:
                logging.critical(
                    'Cannot load caffe2.python. Error: {0}'.format(str(cpu_e)))
                sys.exit(1)

# libcaffe2_python contains a global Workspace that we need to properly delete
# when exiting. Otherwise, cudart will cause segfaults sometimes.
atexit.register(on_module_exit)  # noqa


# Add functionalities for the TensorCPU interface.
def _TensorCPU_shape(self):
    return tuple(self._shape)


def _TensorCPU_reshape(self, shape):
    return self._reshape(list(shape))

TensorCPU.shape = property(_TensorCPU_shape)  # noqa
TensorCPU.reshape = _TensorCPU_reshape  # noqa
