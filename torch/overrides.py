"""
Python implementation of ``__torch_function__``

While most of the torch API and handling for ``__torch_function__`` happens
at the C++ level, some of the torch API is written in Python so we need
python-level handling for ``__torch_function__`` overrides as well. The main
developer-facing functionality in this file are handle_torch_function and
has_torch_function. See torch/functional.py and test/test_overrides.py
for usage examples.

Note
----
heavily inspired by NumPy's ``__array_function__`` (see:
https://github.com/pytorch/pytorch/issues/24015 and
https://www.numpy.org/neps/nep-0018-array-function-protocol.html
)

If changing this file in a way that can affect ``__torch_function__`` overhead,
please report the benchmarks in ``benchmarks/overrides_benchmark``. See the
instructions in the ``README.md`` in that directory.
"""

import __future__

import collections
import functools
import types
from typing import Dict, Set, List, Any, Callable, Iterable

import torch
from torch._C import _is_torch_function_enabled, _disabled_torch_function_impl

__all__ = [
    "get_ignored_functions",
    "get_overridable_functions",
    "get_testing_overrides",
    "handle_torch_function",
    "has_torch_function",
    "is_tensor_like",
    "is_tensor_method_or_property",
    "wrap_torch_function",
]

@functools.lru_cache(None)
def get_ignored_functions() -> Set[Callable]:
    """
    Return public functions that cannot be overridden by ``__torch_function__``.

    Returns
    -------
    Tuple[Callable]
        A tuple of functions that are publicly available in the torch API but cannot
        be overridden with ``__torch_function__``. Mostly this is because none of the
        arguments of these functions are tensors or tensor-likes.

    Examples
    --------
    >>> torch.Tensor.as_subclass in torch.overrides.get_ignored_functions()
    True
    >>> torch.add in torch.overrides.get_ignored_functions()
    False
    """
    Tensor = torch.Tensor
    return {
        torch.typename,
        torch.is_tensor,
        torch.is_storage,
        torch.set_default_tensor_type,
        torch.set_rng_state,
        torch.get_rng_state,
        torch.manual_seed,
        torch.initial_seed,
        torch.seed,
        torch.save,
        torch.load,
        torch.set_printoptions,
        torch.fork,
        torch.get_default_dtype,
        torch.get_num_interop_threads,
        torch.get_num_threads,
        torch.init_num_threads,
        torch.import_ir_module,
        torch.import_ir_module_from_buffer,
        torch.is_anomaly_enabled,
        torch.is_grad_enabled,
        torch.merge_type_from_type_comment,
        torch.parse_ir,
        torch.parse_schema,
        torch.parse_type_comment,
        torch.set_anomaly_enabled,
        torch.set_flush_denormal,
        torch.set_num_interop_threads,
        torch.set_num_threads,
        torch.wait,
        torch.as_tensor,
        torch.from_numpy,
        torch.get_device,
        torch.tensor,
        torch.default_generator,
        torch.has_cuda,
        torch.has_cudnn,
        torch.has_lapack,
        torch.cpp,
        torch.device,
        torch.dtype,
        torch.finfo,
        torch.has_mkl,
        torch.has_mkldnn,
        torch.has_openmp,
        torch.iinfo,
        torch.memory_format,
        torch.qscheme,
        torch.set_grad_enabled,
        torch.no_grad,
        torch.enable_grad,
        torch.layout,
        torch.align_tensors,
        torch.arange,
        torch.as_strided,
        torch.bartlett_window,
        torch.blackman_window,
        torch.broadcast_shapes,
        torch.can_cast,
        torch.cudnn_affine_grid_generator,
        torch.cudnn_batch_norm,
        torch.cudnn_convolution,
        torch.cudnn_convolution_transpose,
        torch.cudnn_grid_sampler,
        torch.cudnn_is_acceptable,
        torch.empty,
        torch.empty_meta,
        torch.empty_strided,
        torch.empty_quantized,
        torch.eye,
        torch.fft.fftfreq,
        torch.fft.rfftfreq,
        torch.from_file,
        torch.full,
        torch.hamming_window,
        torch.hann_window,
        torch.kaiser_window,
        torch.linspace,
        torch.logspace,
        torch.mkldnn_adaptive_avg_pool2d,
        torch.mkldnn_convolution,
        torch.mkldnn_convolution_backward_weights,
        torch.mkldnn_max_pool2d,
        torch.mkldnn_max_pool3d,
        torch.normal,
        torch.ones,
        torch.promote_types,
        torch.rand,
        torch.randn,
        torch.randint,
        torch.randperm,
        torch.range,
        torch.result_type,
        torch.scalar_tensor,
        torch.sparse_coo_tensor,
        torch.tril_indices,
        torch.triu_indices,
        torch.vander,
        torch.zeros,
        torch.nn.functional.assert_int_or_pair,
        torch.nn.functional.boolean_dispatch,
        torch.nn.functional.upsample,
        torch.nn.functional.upsample_bilinear,
        torch.nn.functional.upsample_nearest,
        torch.nn.functional.has_torch_function,
        torch.nn.functional.handle_torch_function,
        torch.nn.functional.sigmoid,
        torch.nn.functional.hardsigmoid,
        torch.nn.functional.tanh,
        torch.set_autocast_enabled,
        torch.is_autocast_enabled,
        torch.clear_autocast_cache,
        torch.autocast_increment_nesting,
        torch.autocast_decrement_nesting,
        torch.nn.functional.hardswish,
        torch.is_vulkan_available,
        torch.is_deterministic,
        torch.set_deterministic,
        torch.unify_type_list,
        Tensor.__delitem__,
        Tensor.__dir__,
        Tensor.__getattribute__,
        Tensor.__init__,
        Tensor.__init_subclass__,
        Tensor.__delattr__,
        Tensor.__setattr__,
        Tensor.__torch_function__,
        Tensor.__new__,
        Tensor.__class__,
        Tensor.__subclasshook__,
        Tensor.as_subclass,
        Tensor.reinforce,
        Tensor.new,
        Tensor.new_tensor,
        Tensor.new_empty,
        Tensor.new_empty_strided,
        Tensor.new_zeros,
        Tensor.new_ones,
        Tensor.new_full,
        Tensor._make_subclass,
        Tensor.stride,
        Tensor.unflatten,
    }


@functools.lru_cache(None)
def get_testing_overrides() -> Dict[Callable, Callable]:
    """Return a dict containing dummy overrides for all overridable functions

    Returns
    -------
    Dict[Callable, Callable]
        A dictionary that maps overridable functions in the PyTorch API to
        lambda functions that have the same signature as the real function
        and unconditionally return -1. These lambda functions are useful
        for testing API coverage for a type that defines ``__torch_function__``.

    Examples
    --------
    >>> import inspect
    >>> my_add = torch.overrides.get_testing_overrides()[torch.add]
    >>> inspect.signature(my_add)
    <Signature (input, other, out=None)>
    """
    # Every function in the PyTorchAPI that can be overriden needs an entry
    # in this dict.
    #
    # Optimally we would use inspect to get the function signature and define
    # the lambda function procedurally but that is blocked by generating
    # function signatures for native kernels that can be consumed by inspect.
    # See Issue #28233.
    Tensor = torch.Tensor
    ret = {
        torch.abs: lambda input, out=None: -1,
        torch.absolute: lambda input, out=None: -1,
        torch.adaptive_avg_pool1d: lambda input, output_size: -1,
        torch.adaptive_max_pool1d: lambda inputs, output_size: -1,
        torch.acos: lambda input, out=None: -1,
        torch.arccos: lambda input, out=None: -1,
        torch.acosh: lambda input, out=None: -1,
        torch.arccosh: lambda input, out=None: -1,
        torch.add: lambda input, other, out=None: -1,
        torch.addbmm: lambda input, batch1, batch2, alpha=1, beta=1, out=None: -1,
        torch.addcdiv: lambda input, tensor1, tensor2, value=1, out=None: -1,
        torch.addcmul: lambda input, tensor1, tensor2, value=1, out=None: -1,
        torch.addmm: lambda input, mat1, mat2, beta=1, alpha=1, out=None: -1,
        torch.addmv: lambda input, mat, vec, beta=1, alpha=1, out=None: -1,
        torch.addr: lambda input, vec1, vec2, beta=1, alpha=1, out=None: -1,
        torch.affine_grid_generator: lambda theta, size, align_corners: -1,
        torch.all: lambda input, dim=None: -1,
        torch.allclose: lambda input, other, trol=1e-05, atol=1e-08, equal_nan=False: -1,
        torch.alpha_dropout: lambda input, p, train, inplace=False: -1,
        torch.amax: lambda input, dim=None: -1,
        torch.amin: lambda input, dim=None: -1,
        torch.angle: lambda input, out=None: -1,
        torch.any: lambda input, dim=None, keepdim=False, out=None: -1,
        torch.argmax: lambda input: -1,
        torch.argmin: lambda input: -1,
        torch.argsort: lambda input, dim=None: -1,
        torch.asin: lambda input, out=None: -1,
        torch.arcsin: lambda input, out=None: -1,
        torch.asinh: lambda input, out=None: -1,
        torch.arcsinh: lambda input, out=None: -1,
        torch.atan: lambda input, out=None: -1,
        torch.arctan: lambda input, out=None: -1,
        torch.atan2: lambda input, other, out=None: -1,
        torch.atanh: lambda input, out=None: -1,
        torch.arctanh: lambda input, out=None: -1,
        torch.atleast_1d: lambda *tensors: -1,
        torch.atleast_2d: lambda *tensors: -1,
        torch.atleast_3d: lambda *tensors: -1,
        torch.avg_pool1d: lambda input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True: -1,
        torch.baddbmm: lambda input, batch1, batch2, alpha=1, beta=1, out=None: -1,
        torch.batch_norm: lambda input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled: -1,
        torch.batch_norm_backward_elemt: lambda grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu: -1,
        torch.batch_norm_backward_reduce: lambda grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g: -1,
        torch.batch_norm_elemt: lambda input, weight, bias, mean, invstd, eps: -1,
        torch.batch_norm_gather_stats: lambda input, mean, invstd, running_mean, running_var, momentum, eps, count: -1,
        torch.batch_norm_gather_stats_with_counts: lambda input, mean, invstd, running_mean, running_var, momentum, eps, count: -1,
        torch.batch_norm_stats: lambda input, eps: -1,
        torch.batch_norm_update_stats: lambda input, running_mean, running_var, momentum: -1,
        torch.bernoulli: lambda input, generator=None, out=None: -1,
        torch.bilinear: lambda input1, input2, weight, bias: -1,
        torch.binary_cross_entropy_with_logits: (lambda input, target, weight=None, size_average=None, reduce=None,
                                                 reduction='mean', pos_weight=None: -1),
        torch.bincount: lambda input, weights=None, minlength=0: -1,
        torch.binomial: lambda count, prob, generator=None: -1,
        torch.bitwise_and: lambda input, other, out=None: -1,
        torch.bitwise_not: lambda input, out=None: -1,
        torch.bitwise_or: lambda input, other, out=None: -1,
        torch.bitwise_xor: lambda input, other, out=None: -1,
        torch.block_diag: lambda *tensors: -1,
        torch.bmm: lambda input, mat2, out=None: -1,
        torch.broadcast_tensors: lambda *tensors: -1,
        torch.bucketize: lambda input, boundaries, out_int32=False, right=False, out=None: -1,
        torch.cartesian_prod: lambda *tensors: -1,
        torch.cat: lambda tensors, dim=0, out=None: -1,
        torch.cdist: lambda x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary': -1,
        torch.ceil: lambda input, out=None: -1,
        torch.celu: lambda input, alhpa=1., inplace=False: -1,
        torch.chain_matmul: lambda *matrices: -1,
        torch.channel_shuffle: lambda input, groups : -1,
        torch.cholesky: lambda input, upper=False, out=None: -1,
        torch.linalg.cholesky: lambda input, out=None: -1,
        torch.cholesky_inverse: lambda input, upper=False, out=None: -1,
        torch.cholesky_solve: lambda input1, input2, upper=False, out=None: -1,
        torch.choose_qparams_optimized: lambda input, numel, n_bins, ratio, bit_width: -1,
        torch.chunk: lambda input, chunks, dim=0: -1,
        torch.clamp: lambda input, min=None, max=None, out=None: -1,
        torch.clip: lambda input, min=None, max=None, out=None: -1,
        torch.clamp_min: lambda input, min, out=None: -1,
        torch.clamp_max: lambda input, max, out=None: -1,
        torch.column_stack: lambda tensors, out=None: -1,
        torch.clone: lambda input: -1,
        torch.combinations: lambda input, r=2, with_replacement=False: -1,
        torch.complex: lambda real, imag: -1,
        torch.copysign: lambda input, other, out=None: -1,
        torch.polar: lambda abs, ang: -1,
        torch.linalg.cond: lambda input, ord=None: -1,
        torch.conj: lambda input, out=None: -1,
        torch.constant_pad_nd: lambda input, pad, value=0: -1,
        torch.conv1d: lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: -1,
        torch.conv2d: lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: -1,
        torch.conv3d: lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: -1,
        torch.convolution: lambda input, weight, bias, stride, padding, dilation, transposed, output_adding, groups: -1,
        torch.conv_tbc: lambda input, weight, bias, pad=0: -1,
        torch.conv_transpose1d: lambda input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: -1,
        torch.conv_transpose2d: lambda input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: -1,
        torch.conv_transpose3d: lambda input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: -1,
        torch.cos: lambda input, out=None: -1,
        torch.cosine_embedding_loss: lambda input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean': -1,
        torch.cosh: lambda input, out=None: -1,
        torch.cosine_similarity: lambda x1, x2, dim=1, eps=1e-8: -1,
        torch.count_nonzero: lambda input: -1,
        torch.cross: lambda input, other, dim=-1, out=None: -1,
        torch.ctc_loss: (lambda log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean',
                         zero_infinity=False: -1),
        torch.cummax: lambda input, dim, out=None: -1,
        torch.cummin: lambda input, dim, out=None: -1,
        torch.cumprod: lambda input, dim, out=None, dtype=None: -1,
        torch.cumsum: lambda input, dim, out=None, dtype=None: -1,
        torch.logcumsumexp: lambda input, dim, out=None: -1,
        torch.deg2rad: lambda input, out=None: -1,
        torch.dequantize: lambda input: -1,
        torch.det: lambda input: -1,
        torch.linalg.det: lambda input: -1,  # alias for torch.det
        torch.detach: lambda input: -1,
        torch.diag: lambda input, diagonal=0, out=None: -1,
        torch.diag_embed: lambda input, diagonal=0, out=None: -1,
        torch.diagflat: lambda input, offset=0: -1,
        torch.diagonal: lambda input, offset=0, dim1=0, dim2=1: -1,
        torch.digamma: lambda input, out=None: -1,
        torch.dist: lambda input, other, p=2: -1,
        torch.div: lambda input, other, out=None: -1,
        torch.divide: lambda input, other, out=None: -1,
        torch.dot: lambda input, other, out=None: -1,
        torch.dropout: lambda input, p, train, inplace=False: -1,
        torch.dsmm: lambda input, mat2: -1,
        torch.hsmm: lambda mat1, mat2: -1,
        torch.dstack: lambda tensors, out=None: -1,
        torch.eig: lambda input, eigenvectors=False, out=None: -1,
        torch.linalg.eigh: lambda input, UPLO="L", out=None: -1,
        torch.linalg.eigvalsh: lambda input, UPLO="L", out=None: -1,
        torch.einsum: lambda equation, *operands: -1,
        torch.embedding: (lambda input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                          sparse=False: -1),
        torch.embedding_bag: (lambda input, weight, offsets, max_norm=None, norm_type=2, scale_grad_by_freq=False,
                              mode='mean', sparse=False, per_sample_weights=None: -1),
        torch.empty_like: lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1,
        torch.eq: lambda input, other, out=None: -1,
        torch.equal: lambda input, other: -1,
        torch.erf: lambda input, out=None: -1,
        torch.erfc: lambda input, out=None: -1,
        torch.erfinv: lambda input, out=None: -1,
        torch.exp: lambda input, out=None: -1,
        torch.exp2: lambda input, out=None: -1,
        torch.expm1: lambda input, out=None: -1,
        torch.fake_quantize_per_channel_affine: lambda input, scale, zero_point, axis, quant_min, quant_max: -1,
        torch.fake_quantize_per_tensor_affine: lambda input, scale, zero_point, quant_min, quant_max: -1,
        torch.fbgemm_linear_fp16_weight: lambda input, packed_weight, bias: -1,
        torch.fbgemm_linear_fp16_weight_fp32_activation: lambda input, packed_weight, bias: -1,
        torch.fbgemm_linear_int8_weight: lambda input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias: -1,
        torch.fbgemm_linear_int8_weight_fp32_activation: (lambda input, weight, packed, col_offsets, weight_scale,
                                                          weight_zero_point, bias: -1),
        torch.fbgemm_linear_quantize_weight: lambda input: -1,
        torch.fbgemm_pack_gemm_matrix_fp16: lambda input: -1,
        torch.fbgemm_pack_quantized_matrix: lambda input, a, b: -1,
        torch.feature_alpha_dropout: lambda input, p, train: -1,
        torch.feature_dropout: lambda input, p, train: -1,
        torch.fft.fft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fft.ifft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fft.rfft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fft.irfft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fft.hfft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fft.ihfft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fft.fftn: lambda input, s=None, dim=None, norm=None: -1,
        torch.fft.ifftn: lambda input, s=None, dim=None, norm=None: -1,
        torch.fft.rfftn: lambda input, s=None, dim=None, norm=None: -1,
        torch.fft.irfftn: lambda input, s=None, dim=None, norm=None: -1,
        torch.fft.fft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        torch.fft.ifft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        torch.fft.rfft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        torch.fft.irfft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        torch.fft.fftshift: lambda input, dim=None: -1,
        torch.fft.ifftshift: lambda input, dim=None: -1,
        torch.fft.fft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fix: lambda input, out=None: -1,
        torch.flatten: lambda input, start_dim=0, end_dim=-1: -1,
        torch.flip: lambda input, dims: -1,
        torch.fliplr: lambda input: -1,
        torch.flipud: lambda input: -1,
        torch.frobenius_norm: lambda input, dim=None, keepdim=False, out=None: -1,
        torch.floor: lambda input, out=None: -1,
        torch.floor_divide: lambda input, other: -1,
        torch.float_power: lambda input, exponent, out=None: -1,
        torch.fmod: lambda input, other, out=None: -1,
        torch.frac: lambda input, out=None: -1,
        torch.full_like: lambda input, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False: -1,
        torch.functional.lu_unpack: lambda LU_data, LU_pivots, unpack_data=True, unpack_pivots=True: -1,
        torch.gather: lambda input, dim, index, out=None, sparse_grad=False: -1,
        torch.gcd: lambda input, other, out=None: -1,
        torch.ge: lambda input, other, out=None: -1,
        torch.greater_equal: lambda input, other, out=None: -1,
        torch.geqrf: lambda input, out=None: -1,
        torch.i0: lambda input, out=None: -1,
        torch.inner: lambda input, other, out=None: -1,
        torch.outer: lambda input, vec2, out=None: -1,  # alias for torch.ger
        torch.ger: lambda input, vec2, out=None: -1,
        torch.grid_sampler: lambda input, grid, interpolation_mode, padding_mode, align_corners: -1,
        torch.grid_sampler_2d: lambda input, grid, interpolation_mode, padding_mode, align_corners: -1,
        torch.grid_sampler_3d: lambda input, grid, interpolation_mode, padding_mode, align_corners: -1,
        torch.group_norm: lambda input, num_groups, weight=None, bias=None, eps=1e-05, cudnn_enabled=True: -1,
        torch.gru: lambda input, hx, params, has_biases, num_layers, gropout, train, bidirectional, batch_first: -1,
        torch.gru_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        torch.gt: lambda input, other, out=None: -1,
        torch.greater: lambda input, other, out=None: -1,
        torch.hardshrink: lambda input, lambd=0.5: -1,
        torch.heaviside: lambda input, values, out=None: -1,
        torch.hinge_embedding_loss: lambda input, target, margin=1.0, size_average=None, reduce=None, reduction='mean': -1,
        torch.histc: lambda input, bins=100, min=0, max=0, out=None: -1,
        torch.hspmm: lambda mat1, mat2, out=None: -1,
        torch.hstack: lambda tensors, out=None: -1,
        torch.hypot: lambda input, other, out=None: -1,
        torch.igamma: lambda input, other, out=None: -1,
        torch.igammac: lambda input, other, out=None: -1,
        torch.imag: lambda input, out=None: -1,
        torch.index_add: lambda input, dim, index, source: -1,
        torch.index_copy: lambda input, dim, index, source: -1,
        torch.index_put: lambda input, indices, values, accumulate=False: -1,
        torch.index_select: lambda input, dim, index, out=None: -1,
        torch.index_fill: lambda input, dim, index, value: -1,
        torch.isfinite: lambda tensor: -1,
        torch.isinf: lambda tensor: -1,
        torch.isreal: lambda tensor: -1,
        torch.isposinf: lambda input, out=None: -1,
        torch.isneginf: lambda input, out=None: -1,
        torch.instance_norm: (lambda input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps,
                              cudnn_enabled: -1),
        torch.int_repr: lambda input: -1,
        torch.inverse: lambda input, out=None: -1,
        torch.is_complex: lambda input: -1,
        torch.is_distributed: lambda input: -1,
        torch.is_floating_point: lambda input: -1,
        torch.is_nonzero: lambda input: -1,
        torch.is_same_size: lambda input, other: -1,
        torch.is_signed: lambda input: -1,
        torch.isclose: lambda input, other, rtol=1e-05, atol=1e-08, equal_nan=False: -1,
        torch.isnan: lambda input: -1,
        torch.istft: (lambda input, n_fft, hop_length=None, win_length=None, window=None, center=True,
                      normalized=False, onesided=None, length=None, return_complex=False: -1),
        torch.kl_div: lambda input, target, size_average=None, reduce=None, reduction='mean', log_target=False: -1,
        torch.kron: lambda input, other: -1,
        torch.kthvalue: lambda input, k, dim=None, keepdim=False, out=None: -1,
        torch.layer_norm: lambda input, normalized_shape, weight=None, bias=None, esp=1e-05, cudnn_enabled=True: -1,
        torch.lcm: lambda input, other, out=None: -1,
        torch.ldexp: lambda input, other, out=None: -1,
        torch.le: lambda input, other, out=None: -1,
        torch.less_equal: lambda input, other, out=None: -1,
        torch.lerp: lambda input, end, weight, out=None: -1,
        torch.lgamma: lambda input, out=None: -1,
        torch.lobpcg: lambda input, k=None, B=None, X=None, n=None, iK=None, niter=None, tol=None, largest=None, method=None,
        tracker=None, ortho_iparams=None, ortho_fparams=None, ortho_bparams=None: -1,
        torch.log: lambda input, out=None: -1,
        torch.log_softmax: lambda input, dim, dtype=None: -1,
        torch.log10: lambda input, out=None: -1,
        torch.log1p: lambda input, out=None: -1,
        torch.log2: lambda input, out=None: -1,
        torch.logaddexp: lambda input, other, out=None: -1,
        torch.logaddexp2: lambda input, other, out=None: -1,
        torch.logdet: lambda input: -1,
        torch.logical_and: lambda input, other, out=None: -1,
        torch.logical_not: lambda input, out=None: -1,
        torch.logical_or: lambda input, other, out=None: -1,
        torch.logical_xor: lambda input, other, out=None: -1,
        torch.logsumexp: lambda input, names, keepdim=False, out=None: -1,
        torch.logit: lambda input, eps=None: -1,
        torch.logsumexp: lambda input, names, keepdim=False, out=None: -1,
        torch.lstm: lambda data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional: -1,
        torch.lstm_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        torch.lstsq: lambda input, A, out=None: -1,
        torch.lt: lambda input, other, out=None: -1,
        torch.less: lambda input, other, out=None: -1,
        torch.lu: lambda A, pivot=True, get_infos=False, out=None: -1,
        torch.lu_solve: lambda b, LU_data, LU_pivots, out=None: -1,
        torch.margin_ranking_loss: lambda input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean': -1,
        torch.masked_fill: lambda input, mask, value: -1,
        torch.masked_scatter: lambda input, mask, source: -1,
        torch.masked_select: lambda input, mask, out=None: -1,
        torch.matmul: lambda input, other, out=None: -1,
        torch.matrix_power: lambda input, n: -1,
        torch.matrix_rank: lambda input, tol=None, symmetric=False: -1,
        torch.linalg.matrix_rank: lambda input, tol=None, hermitian=False: -1,
        torch.matrix_exp: lambda input: -1,
        torch.max: lambda input, out=None: -1,
        torch.maximum: lambda input, other, out=None: -1,
        torch.max_pool1d: lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False: -1,
        torch.max_pool2d: lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False: -1,
        torch.max_pool3d: lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False: -1,
        torch.max_pool1d_with_indices: (lambda input, kernel_size, stride=None, padding=0, dilation=1,
                                        return_indices=False, ceil_mode=False: -1),
        torch.mean: lambda input, dim=None: -1,
        torch.median: lambda input, dim=None: -1,
        torch.nanmedian: lambda input, dim=None: -1,
        torch.meshgrid: lambda *tensors, **kwargs: -1,
        torch.min: lambda input, out=None: -1,
        torch.minimum: lambda input, other, out=None: -1,
        torch.miopen_batch_norm: (lambda input, weight, bias, running_mean, running_var, training,
                                  exponential_average_factor, epsilon: -1),
        torch.miopen_convolution: lambda input, weight, bias, padding, stride, dilation, groups, benchmark, deterministic: -1,
        torch.miopen_convolution_transpose: (lambda input, weight, bias, padding, output_padding, stride, dilation,
                                             groups, benchmark, deterministic: -1),
        torch.miopen_depthwise_convolution: (lambda input, weight, bias, padding, stride, dilation, groups, benchmark,
                                             deterministic: -1),
        torch.miopen_rnn: (lambda input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first,
                           dropout, train, bidirectional, batch_sizes, dropout_state: -1),
        torch.mm: lambda input, mat2, out=None: -1,
        torch.mode: lambda input, dim=-1, keepdim=False, out=None: -1,
        torch.movedim: lambda input, source, destination: -1,
        torch.moveaxis: lambda input, source, destination: -1,
        torch.msort: lambda input, descending=False, out=None: -1,
        torch.mul: lambda input, other, out=None: -1,
        torch.multiply: lambda input, other, out=None: -1,
        torch.multinomial: lambda input, num_samples, replacement=False, out=None: -1,
        torch.mv: lambda input, vec, out=None: -1,
        torch.mvlgamma: lambda input, p: -1,
        torch.narrow: lambda input, dim, start, length: -1,
        torch.nan_to_num: lambda input, nan=0.0, posinf=None, neginf=None, out=None: -1,
        torch.native_batch_norm: lambda input, weight, bias, running_mean, running_var, training, momentum, eps: -1,
        torch.native_layer_norm: lambda input, weight, bias, M, N, eps: -1,
        torch.native_group_norm: lambda input, weight, bias, N, C, HxW, group, eps: -1,
        torch.native_norm: lambda input, p=2: -1,
        torch.native_norm: lambda input, p=2: -1,
        torch.native_norm: lambda input, p=2, dim=None, keepdim=False, dtype=None: -1,
        torch.ne: lambda input, other, out=None: -1,
        torch.not_equal: lambda input, other, out=None: -1,
        torch.neg: lambda input, out=None: -1,
        torch.negative: lambda input, out=None: -1,
        torch.nextafter: lambda input, other, out=None: -1,
        torch.nn.functional.adaptive_avg_pool2d: lambda input, output_size: -1,
        torch.nn.functional.adaptive_avg_pool3d: lambda input, output_size: -1,
        torch.nn.functional.adaptive_max_pool1d: lambda input, output_size, return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool1d_with_indices: lambda input, output_size, return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool2d: lambda input, output_size, return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool2d_with_indices: lambda input, output_size, return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool3d: lambda input, output_size, return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool3d_with_indices: lambda input, output_size, return_indices=False: -1,
        torch.nn.functional.affine_grid: lambda theta, size, align_corners=None: -1,
        torch.nn.functional.alpha_dropout: lambda input, p=0.5, training=False, inplace=False: -1,
        torch.nn.functional.avg_pool2d: (lambda input, kernel_size, stride=None, padding=0, ceil_mode=False,
                                         count_include_pad=True, divisor_override=None: -1),
        torch.nn.functional.avg_pool3d: (lambda input, kernel_size, stride=None, padding=0, ceil_mode=False,
                                         count_include_pad=True, divisor_override=None: -1),
        torch.nn.functional.batch_norm: (lambda input, running_mean, running_var, weight=None, bias=None, training=False,
                                         momentum=0.1, eps=1e-05: -1),
        torch.nn.functional.bilinear: lambda input1, input2, weight, bias=None: -1,
        torch.nn.functional.binary_cross_entropy: (lambda input, target, weight=None, size_average=None, reduce=None,
                                                   reduction="mean": -1),
        torch.nn.functional.binary_cross_entropy_with_logits: (lambda input, target, weight=None, size_average=None,
                                                               reduce=None, reduction="mean", pos_weight=None: -1),
        torch.nn.functional.celu: lambda input, alpha=1.0, inplace=False: -1,
        torch.nn.functional.cosine_embedding_loss: (lambda input1, input2, target, margin=0, size_average=None,
                                                    reduce=None, reduction='mean': -1),
        torch.nn.functional.cross_entropy: (lambda input, target, weight=None, size_average=None, ignore_index=-100,
                                            reduce=None, reduction="mean": -1),
        torch.nn.functional.ctc_loss: (lambda log_probs, targets, input_lengths, target_lengths, blank=0,
                                       reduction='mean', zero_infinity=False: -1),
        torch.nn.functional.dropout: lambda input, p=0.5, training=True, inplace=False: -1,
        torch.nn.functional.dropout2d: lambda input, p=0.5, training=True, inplace=False: -1,
        torch.nn.functional.dropout3d: lambda input, p=0.5, training=True, inplace=False: -1,
        torch.nn.functional.elu: lambda input, alpha=1.0, inplace=False: -1,
        torch.nn.functional.embedding: (lambda input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
                                        scale_grad_by_freq=False, sparse=False: -1),
        torch.nn.functional.embedding_bag: (lambda input, weight, offsets=None, max_norm=None, norm_type=2,
                                            scale_grad_by_freq=False, mode='mean', sparse=False, per_sample_weights=None,
                                            include_last_offset=False: -1),
        torch.nn.functional.feature_alpha_dropout: lambda input, p=0.5, training=False, inplace=False: -1,
        torch.nn.functional.fold: lambda input, output_size, kernel_size, dilation=1, padding=0, stride=1: -1,
        torch.nn.functional.fractional_max_pool2d: (lambda input, kernel_size, output_size=None, output_ratio=None,
                                                    return_indices=False, _random_samples=None: -1),
        torch.nn.functional.fractional_max_pool2d_with_indices: (
            lambda input, kernel_size, output_size=None, output_ratio=None, return_indices=False,
            _random_samples=None: -1),
        torch.nn.functional.fractional_max_pool3d: (lambda input, kernel_size, output_size=None, output_ratio=None,
                                                    return_indices=False, _random_samples=None: -1),
        torch.nn.functional.fractional_max_pool3d_with_indices: (
            lambda input, kernel_size, output_size=None, output_ratio=None, return_indices=False,
            _random_samples=None: -1),
        torch.nn.functional.gelu: lambda input: -1,
        torch.nn.functional.glu: lambda input, dim=-1: -1,
        torch.nn.functional.grid_sample: lambda input, grid, mode='bilinear', padding_mode='zeros', align_corners=None: -1,
        torch.nn.functional.group_norm: lambda input, num_groups, weight=None, bias=None, eps=1e-05: -1,
        torch.nn.functional.gumbel_softmax: lambda logits, tau=1, hard=False, eps=1e-10, dim=-1: -1,
        torch.nn.functional.hardshrink: lambda input, lambd=0.5: -1,
        torch.nn.functional.hardtanh: lambda input, min_val=-1., max_val=1., inplace=False: -1,
        torch.nn.functional.hinge_embedding_loss: (lambda input, target, margin=1.0, size_average=None, reduce=None,
                                                   reduction='mean': -1),
        torch.nn.functional.instance_norm: (lambda input, running_mean=None, running_var=None, weight=None, bias=None,
                                            use_input_stats=True, momentum=0.1, eps=1e-05: -1),
        torch.nn.functional.interpolate: (lambda input, size=None, scale_factor=None, mode='nearest', align_corners=None,
                                          recompute_scale_factor=None: -1),
        torch.nn.functional.kl_div: lambda input, target, size_average=None, reduce=None, reduction='mean', log_target=False: -1,
        torch.nn.functional.l1_loss: lambda input, target, size_average=None, reduce=None, reduction='mean': -1,
        torch.nn.functional.layer_norm: lambda input, normalized_shape, weight=None, bias=None, eps=1e-05: -1,
        torch.nn.functional.leaky_relu: lambda input, negative_slope=0.01, inplace=False: -1,
        torch.nn.functional.linear: lambda input, weight, bias=None: -1,
        torch.nn.functional.local_response_norm: lambda input, size, alpha=0.0001, beta=0.75, k=1.0: -1,
        torch.nn.functional.log_softmax: lambda input, dim=None, _stacklevel=3, dtype=None: -1,
        torch.nn.functional.logsigmoid: lambda input: -1,
        torch.nn.functional.lp_pool1d: lambda input, norm_type, kernel_size, stride=None, ceil_mode=False: -1,
        torch.nn.functional.lp_pool2d: lambda input, norm_type, kernel_size, stride=None, ceil_mode=False: -1,
        torch.nn.functional.margin_ranking_loss: (lambda input1, input2, target, margin=0, size_average=None,
                                                  reduce=None, reduction='mean': -1),
        torch.nn.functional.max_pool1d: (lambda input, kernel_size, stride=None, padding=0, dilation=1,
                                         ceil_mode=False, return_indices=False: -1),
        torch.nn.functional.max_pool1d_with_indices: (lambda input, kernel_size, stride=None, padding=0, dilation=1,
                                                      return_indices=False, ceil_mode=False: -1),
        torch.nn.functional.max_pool2d: (lambda input, kernel_size, stride=None, padding=0, dilation=1,
                                         ceil_mode=False, return_indices=False: -1),
        torch.nn.functional.max_pool2d_with_indices: (lambda input, kernel_size, stride=None, padding=0, dilation=1,
                                                      return_indices=False, ceil_mode=False: -1),
        torch.nn.functional.max_pool3d: (lambda input, kernel_size, stride=None, padding=0, dilation=1,
                                         return_indices=False, ceil_mode=False: -1),
        torch.nn.functional.max_pool3d_with_indices: (lambda input, kernel_size, stride=None, padding=0, dilation=1,
                                                      return_indices=False, ceil_mode=False: -1),
        torch.nn.functional.max_unpool1d: lambda input, indices, kernel_size, stride=None, padding=0, output_size=None: -1,
        torch.nn.functional.max_unpool2d: lambda input, indices, kernel_size, stride=None, padding=0, output_size=None: -1,
        torch.nn.functional.max_unpool3d: lambda input, indices, kernel_size, stride=None, padding=0, output_size=None: -1,
        torch.nn.functional.mse_loss: lambda input, target, size_average=None, reduce=None, reduction='mean': -1,
        torch.nn.functional.multi_head_attention_forward: (
            lambda query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v,
            add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None,
            need_weights=True, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
            v_proj_weight=None, static_k=None, static_v=None: -1),
        torch.nn.functional.multi_margin_loss: (lambda input, target, p=1, margin=1.0, weight=None, size_average=None,
                                                reduce=None, reduction='mean': -1),
        torch.nn.functional.multilabel_margin_loss: (lambda input, target, size_average=None, reduce=None,
                                                     reduction='mean': -1),
        torch.nn.functional.multilabel_soft_margin_loss: (lambda input, target, weight=None, size_average=None,
                                                          reduce=None, reduction='mean': -1),
        torch.nn.functional.nll_loss: (lambda input, target, weight=None, size_average=None, ignore_index=-100,
                                       reduce=None, reduction='mean': -1),
        torch.nn.functional.normalize: lambda input, p=2, dim=1, eps=1e-12, out=None: -1,
        torch.nn.functional.one_hot: lambda tensor, num_classes=-1: -1,
        torch.nn.functional.pad: lambda input, pad, mode='constant', value=0: -1,
        torch.nn.functional.pairwise_distance: lambda x1, x2, p=2.0, eps=1e-06, keepdim=False: -1,
        torch.nn.functional.poisson_nll_loss: (lambda input, target, log_input=True, full=False, size_average=None,
                                               eps=1e-08, reduce=None, reduction='mean': -1),
        torch.nn.functional.prelu: lambda input, weight: -1,
        torch.nn.functional.relu: lambda input, inplace=False: -1,
        torch.nn.functional.relu6: lambda input, inplace=False: -1,
        torch.nn.functional.rrelu: lambda input, lower=0.125, upper=0.3333333333333333, training=False, inplace=False: -1,
        torch.nn.functional.selu: lambda input, inplace=False: -1,
        torch.nn.functional.silu: lambda input, inplace=False: -1,
        torch.nn.functional.smooth_l1_loss: lambda input, target, size_average=None, reduce=None, reduction='mean', beta=1.: -1,
        torch.nn.functional.soft_margin_loss: lambda input, target, size_average=None, reduce=None, reduction='mean': -1,
        torch.nn.functional.softmax: lambda input, dim=None, _stacklevel=3, dtype=None: -1,
        torch.nn.functional.softmin: lambda input, dim=None, _stacklevel=3, dtype=None: -1,
        torch.nn.functional.softplus: lambda input, beta=1, threshold=20: -1,
        torch.nn.functional.softshrink: lambda input, lambd=0.5: -1,
        torch.nn.functional.softsign: lambda input: -1,
        torch.nn.functional.tanhshrink: lambda input: -1,
        torch.nn.functional.threshold: lambda input, threshold, value, inplace=False: -1,
        torch.nn.functional.triplet_margin_loss: (lambda anchor, positive, negative, margin=1.0, p=2, eps=1e-06,
                                                  swap=False, size_average=None, reduce=None, reduction='mean': -1),
        torch.nn.functional.triplet_margin_with_distance_loss: (lambda anchor, positive, negative, *,
                                                                distance_function=None, margin=1.0,
                                                                swap=False, reduction='mean': -1),
        torch.nn.functional.unfold: lambda input, kernel_size, dilation=1, padding=0, stride=1: -1,
        torch.nonzero: lambda input, as_tuple=False: -1,
        torch.norm: lambda input, p='fro', dim=None, keepdim=False, out=None, dtype=None: -1,
        torch.norm_except_dim: lambda v, pow=2, dim=0: -1,
        torch.nuclear_norm: lambda input, p='fro', dim=None, keepdim=False, out=None, dtype=None: -1,
        torch.numel: lambda input: -1,
        torch.orgqr: lambda input1, input2: -1,
        torch.ormqr: lambda input, input2, input3, left=True, transpose=False: -1,
        torch.pairwise_distance: lambda x1, x2, p=2.0, eps=1e-06, keepdim=False: -1,
        torch.pca_lowrank: lambda input, q=None, center=True, niter=2: -1,
        torch.pdist: lambda input, p=2: -1,
        torch.pinverse: lambda input, rcond=1e-15: -1,
        torch.pixel_shuffle: lambda input, upscale_factor: -1,
        torch.poisson: lambda input, generator=None: -1,
        torch.poisson_nll_loss: lambda input, target, log_input, full, eps, reduction: -1,
        torch.polygamma: lambda input, n, out=None: -1,
        torch.prelu: lambda input, weight: -1,
        torch.ones_like: lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1,
        torch.pow: lambda input, exponent, out=None: -1,
        torch.prod: lambda input, dtype=None: -1,
        torch.q_per_channel_axis: lambda input: -1,
        torch.q_per_channel_scales: lambda input: -1,
        torch.q_per_channel_zero_points: lambda input: -1,
        torch.q_scale: lambda input: -1,
        torch.q_zero_point: lambda input: -1,
        torch.qr: lambda input, some=True, out=None: -1,
        torch.quantile: lambda input, q, dim=None, keepdim=False, out=None: -1,
        torch.nanquantile: lambda input, q, dim=None, keepdim=False, out=None: -1,
        torch.quantize_per_channel: lambda input, scales, zero_points, axis, dtype: -1,
        torch.quantize_per_tensor: lambda input, scale, zero_point, dtype: -1,
        torch.quantized_batch_norm: lambda input, weight, bias, mean, var, eps, output_scale, output_zero_point: -1,
        torch.quantized_gru_cell: (lambda input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih,
                                   col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh: -1),

        torch.quantized_lstm_cell: (lambda input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih,
                                    col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh: -1),
        torch.quantized_max_pool1d: (lambda input, kernel_size, stride=tuple(), padding=(0,),
                                     dilation=(1,), ceil_mode=False: -1),
        torch.quantized_max_pool2d: (lambda input, kernel_size, stride=tuple(), padding=(0, 0),
                                     dilation=(1, 1), ceil_mode=False: -1),
        torch.quantized_rnn_relu_cell: (lambda input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih,
                                        col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh: -1),
        torch.quantized_rnn_tanh_cell: (lambda input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih,
                                        col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh: -1),
        torch.rad2deg: lambda input, out=None: -1,
        torch.rand_like: lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1,
        torch.randint_like: lambda input, high, dtype=None, layout=torch.strided, device=None, requires_grad=False: -1,
        torch.randn_like: lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1,
        torch.ravel: lambda input: -1,
        torch.real: lambda input, out=None: -1,
        torch.vdot: lambda input, other, out=None: -1,
        torch.view_as_real: lambda input: -1,
        torch.view_as_complex: lambda input: -1,
        torch.reciprocal: lambda input, out=None: -1,
        torch.relu: lambda input, inplace=False: -1,
        torch.remainder: lambda input, other, out=None: -1,
        torch.renorm: lambda input, p, dim, maxnorm, out=None: -1,
        torch.repeat_interleave: lambda input, dim=None: -1,
        torch.reshape: lambda input, shape: -1,
        torch.rnn_relu: lambda input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first: -1,
        torch.rnn_relu_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        torch.rnn_tanh: lambda input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first: -1,
        torch.rnn_tanh_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        torch.roll: lambda input, shifts, dims=None: -1,
        torch.rot90: lambda input, k=1, dims=(0, 1): -1,
        torch.round: lambda input, out=None: -1,
        torch.row_stack: lambda tensors, out=None: -1,  # alias for torch.vstack
        torch.rowwise_prune: (lambda weight, mask, compressed_indices_dtype: -1),
        torch.rrelu: lambda input, lower=1. / 8, upper=1. / 3, training=False, inplace=False: -1,
        torch.rsqrt: lambda input, out=None: -1,
        torch.rsub: lambda input, other, alpha=1: -1,
        torch.saddmm: lambda input, mat1, mat2, beta=1, alpha=1, out=None: -1,
        torch.scatter: lambda input, dim, index, src: -1,
        torch.scatter_add: lambda input, dim, index, src: -1,
        torch.searchsorted: lambda sorted_sequence, input, out_int32=False, right=False, out=None: -1,
        torch.select: lambda input, dim, index: -1,
        torch.selu: lambda input, inplace=False: -1,
        torch.sigmoid: lambda input, out=None: -1,
        torch.sign: lambda input, out=None: -1,
        torch.signbit: lambda input, out=None: -1,
        torch.sgn: lambda input, out=None: -1,
        torch.sin: lambda input, out=None: -1,
        torch.sinh: lambda input, out=None: -1,
        torch.slogdet: lambda input: -1,
        torch.smm: lambda input, mat2: -1,
        torch.spmm: lambda input, mat2: -1,
        torch.softmax: lambda input, dim, dtype=None: -1,
        torch.solve: lambda input, A, out=None: -1,
        torch.sort: lambda input, dim=-1, descending=False, out=None: -1,
        torch.split: lambda tensor, split_size_or_sections, dim=0: -1,
        torch.split_with_sizes: lambda tensor, split_size_or_sections, dim=0: -1,
        torch.sqrt: lambda input, out=None: -1,
        torch.square: lambda input, out=None: -1,
        torch.squeeze: lambda input, dim=None, out=None: -1,
        torch.sspaddmm: lambda input, mat1, mat2, beta=1, alpha=1, out=None: -1,
        torch.stack: lambda tensors, dim=0, out=None: -1,
        torch.std: lambda input, dim=None: -1,
        torch.std_mean: lambda input, dim=None: -1,
        torch.stft: (lambda input, n_fft, hop_length=None, win_length=None, window=None, center=True,
                     pad_mode='reflect', normalized=False, onesided=True, return_complex=None: -1),
        torch.sub: lambda input, other, out=None: -1,
        torch.subtract: lambda input, other, out=None: -1,
        torch.sum: lambda input, dim=None: -1,
        torch.nansum: lambda input, dim=None: -1,
        torch.svd: lambda input, some=True, compute_uv=True, out=None: -1,
        torch.svd_lowrank: lambda input, q=6, niter=2, M=None: -1,
        torch.symeig: lambda input, eigenvectors=False, upper=True, out=None: -1,
        torch.swapaxes: lambda input, dim0, dim1: -1,
        torch.swapdims: lambda input, axis0, axis1: -1,
        torch.t: lambda input: -1,
        torch.take: lambda input, index: -1,
        torch.tan: lambda input, out=None: -1,
        torch.tanh: lambda input, out=None: -1,
        torch.linalg.tensorinv: lambda a, ind=2: -1,
        torch.linalg.tensorsolve: lambda a, b, dims=None: -1,
        torch.tensordot: lambda a, b, dims=2, out=None: -1,
        torch.tensor_split: lambda input, indices_or_sections, dim=0: -1,
        torch.threshold: lambda input, threshold, value, inplace=False: -1,
        torch.tile: lambda input, dims: -1,
        torch.topk: lambda input, k, dim=-1, descending=False, out=None: -1,
        torch.trace: lambda input: -1,
        torch.transpose: lambda input, dim0, dim1: -1,
        torch.trapz: lambda y, x=None, dim=-1: -1,
        torch.triangular_solve: lambda input, A, upper=True, transpose=False, unitriangular=False: -1,
        torch.tril: lambda input, diagonal=0, out=None: -1,
        torch.triplet_margin_loss: (lambda anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False,
                                    size_average=None, reduce=None, reduction='mean': -1),
        torch.triu: lambda input, diagonal=0, out=None: -1,
        torch.true_divide: lambda input, other: -1,
        torch.trunc: lambda input, out=None: -1,
        torch.unbind: lambda input, dim=0: -1,
        torch.unique: lambda input, sorted=True, return_inverse=False, return_counts=False, dim=None: -1,
        torch.unique_consecutive: lambda input, return_inverse=False, return_counts=False, dim=None: -1,
        torch.unsafe_chunk: lambda input, chunks, dim=0: -1,
        torch.unsafe_split: lambda tensor, split_size_or_sections, dim=0: -1,
        torch.unsafe_split_with_sizes: lambda tensor, split_size_or_sections, dim=0: -1,
        torch.unsqueeze: lambda input, dim, out=None: -1,
        torch.var: lambda input, dim=None: -1,
        torch.var_mean: lambda input, dim=None: -1,
        torch.vstack: lambda tensors, out=None: -1,
        torch.where: lambda condition, x=None, y=None: -1,
        torch.zeros_like: lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1,
        Tensor.__floordiv__: lambda self, other: -1,
        Tensor.__rfloordiv__: lambda self, other: -1,
        Tensor.__ifloordiv__: lambda self, other: -1,
        Tensor.__truediv__: lambda self, other: -1,
        Tensor.__rtruediv__: lambda self, other: -1,
        Tensor.__itruediv__: lambda self, other: -1,
        Tensor.__lshift__: lambda self, other: -1,
        Tensor.__ilshift__: lambda self, other: -1,
        Tensor.__rshift__: lambda self, other: -1,
        Tensor.__irshift__: lambda self, other: -1,
        Tensor.__float__: lambda self: -1,
        Tensor.__complex__: lambda self: -1,
        Tensor.__array__: lambda self, dtype: -1,
        Tensor.__bool__: lambda self: -1,
        Tensor.__contains__: lambda self, other: -1,
        Tensor.__neg__: lambda self: -1,
        Tensor.__invert__: lambda self: -1,
        Tensor.__mod__: lambda self, other: -1,
        Tensor.__array_wrap__: lambda self, array: -1,
        Tensor.__getitem__: lambda self, idx: -1,
        Tensor.__deepcopy__: lambda self, memo: -1,
        Tensor.__iter__: lambda self: -1,
        Tensor.__int__: lambda self: -1,
        Tensor.__long__: lambda self: -1,
        Tensor.__hash__: lambda self: -1,
        Tensor.__index__: lambda self: -1,
        Tensor.__len__: lambda self: -1,
        Tensor.__format__: lambda self, format_spec: -1,
        Tensor.__reduce_ex__: lambda self, proto: -1,
        Tensor.__reversed__: lambda self: -1,
        Tensor.__repr__: lambda self: -1,
        Tensor.__setitem__: lambda self, k, v: -1,
        Tensor.__setstate__: lambda self, d: -1,
        Tensor.T.__get__: lambda self: -1,
        Tensor._backward_hooks.__get__: lambda self: -1,
        Tensor._base.__get__: lambda self: -1,
        Tensor._cdata.__get__: lambda self: -1,
        Tensor.grad.__get__: lambda self: -1,
        Tensor._grad.__get__: lambda self: -1,
        Tensor._grad_fn.__get__: lambda self: -1,
        Tensor.grad_fn.__get__: lambda self: -1,
        Tensor._version.__get__: lambda self: -1,
        Tensor.data.__get__: lambda self: -1,
        Tensor.device.__get__: lambda self: -1,
        Tensor.dtype.__get__: lambda self: -1,
        Tensor.is_cuda.__get__: lambda self: -1,
        Tensor.is_leaf.__get__: lambda self: -1,
        Tensor.is_meta.__get__: lambda self: -1,
        Tensor.is_mkldnn.__get__: lambda self: -1,
        Tensor.is_quantized.__get__: lambda self: -1,
        Tensor.is_sparse.__get__: lambda self: -1,
        Tensor.is_vulkan.__get__: lambda self: -1,
        Tensor.layout.__get__: lambda self: -1,
        Tensor.name.__get__: lambda self: -1,
        Tensor.names.__get__: lambda self: -1,
        Tensor.ndim.__get__: lambda self: -1,
        Tensor.output_nr.__get__: lambda self: -1,
        Tensor.requires_grad.__get__: lambda self: -1,
        Tensor.shape.__get__: lambda self: -1,
        Tensor.volatile.__get__: lambda self: -1,
        Tensor.real.__get__: lambda self: -1,
        Tensor.imag.__get__: lambda self: -1,
        Tensor.__cuda_array_interface__.__get__: lambda self: -1,
        Tensor.type: lambda self, dtype=None, non_blocking=False, **kwargs: -1,
        Tensor._coalesced_: lambda self: -1,
        Tensor._dimI: lambda self: -1,
        Tensor._dimV: lambda self: -1,
        Tensor._indices: lambda self: -1,
        Tensor._is_view: lambda self: -1,
        Tensor._nnz: lambda self: -1,
        Tensor._update_names: lambda self, names, inplace: -1,
        Tensor._values: lambda self: -1,
        Tensor.align_as: lambda self, other: -1,
        Tensor.align_to: lambda self, order, ellipsis_idx: -1,
        Tensor.apply_: lambda self, callable: -1,
        Tensor.as_strided: lambda self, size, stride: -1,
        Tensor.as_strided_: lambda self, size, stride: -1,
        Tensor.backward: lambda self, gradient=None, retain_graph=None, create_graph=False, inputs=None: -1,
        Tensor.bfloat16: lambda self, memory_format=torch.preserve_format: -1,
        Tensor.bool: lambda self, memory_format=torch.preserve_format: -1,
        Tensor.byte: lambda self, memory_format=torch.preserve_format: -1,
        Tensor.char: lambda self, memory_format=torch.preserve_format: -1,
        Tensor.cauchy_: lambda self, median=0, sigma=1, *, generator=None: -1,
        Tensor.coalesce: lambda self: -1,
        Tensor._coalesced_: lambda self, coalesced: -1,
        Tensor.contiguous: lambda self, memory_format=torch.contiguous_format: -1,
        Tensor.copy_: lambda self, src, non_blocking=False: -1,
        Tensor.cpu: lambda self, memory_format=torch.preserve_format: -1,
        Tensor.cuda: lambda self, memory_format=torch.preserve_format: -1,
        Tensor.data_ptr: lambda self: -1,
        Tensor.dense_dim: lambda self: -1,
        Tensor.dim: lambda self: -1,
        Tensor.double: lambda self, memory_format=torch.preserve_format: -1,
        Tensor.element_size: lambda self: -1,
        Tensor.expand: lambda self, size: -1,
        Tensor.expand_as: lambda self, other: -1,
        Tensor.exponential_: lambda self, lambd=1, *, generator=None: -1,
        Tensor.fill_: lambda self, value: -1,
        Tensor.fill_diagonal_: lambda self, value: -1,
        Tensor.float: lambda self, memory_format=torch.preserve_format: -1,
        Tensor.geometric_: lambda self, p, *, generator=None: -1,
        Tensor.get_device: lambda self: -1,
        Tensor.half: lambda self, memory_format=torch.preserve_format: -1,
        Tensor.has_names: lambda self: -1,
        Tensor.indices: lambda self: -1,
        Tensor.int: lambda self, memory_format=torch.preserve_format: -1,
        Tensor.is_coalesced: lambda self: -1,
        Tensor.is_contiguous: lambda self: -1,
        Tensor.is_pinned: lambda self: -1,
        Tensor.is_set_to: lambda self, tensor: -1,
        Tensor.is_shared: lambda self: -1,
        Tensor.item: lambda self: -1,
        Tensor.log_normal_: lambda self, mean=1, std=2, *, generator=None: -1,
        Tensor.log_softmax: lambda self, dim: -1,
        Tensor.long: lambda self, memory_format=torch.preserve_format: -1,
        Tensor.map_: lambda self, tensor, callable: -1,
        Tensor.map2_: lambda self, x, y, callable: -1,
        Tensor.mm: lambda self, mat2: -1,
        Tensor.narrow_copy: lambda self, dimension, start, length: -1,
        Tensor.ndimension: lambda self: -1,
        Tensor.nelement: lambda self: -1,
        Tensor.normal_: lambda self: -1,
        Tensor.numpy: lambda self: -1,
        Tensor.permute: lambda self, dim: -1,
        Tensor.pin_memory: lambda self: -1,
        Tensor.put_: lambda self, indices, tensor, accumulate=False: -1,
        Tensor.qscheme: lambda self: -1,
        Tensor.random_: lambda self, from_=0, to=None, *, generator=None: -1,
        Tensor.record_stream: lambda self, stream: -1,
        Tensor.refine_names: lambda self, names: -1,
        Tensor.register_hook: lambda self, hook: -1,
        Tensor.rename: lambda self, name: -1,
        Tensor.repeat: lambda self, *size: -1,
        Tensor.requires_grad_: lambda self, requires_grad=True: -1,
        Tensor.reshape_as: lambda self, other: -1,
        Tensor.resize: lambda self, *size: -1,
        Tensor.resize_: lambda self, size: -1,
        Tensor.resize_as: lambda self, other: -1,
        Tensor.retain_grad: lambda self: -1,
        Tensor.set_: lambda self, source=None, storage_offset=0, size=None, stride=None: -1,
        Tensor.share_memory_: lambda self: -1,
        Tensor.short: lambda self, memory_format=torch.preserve_format: -1,
        Tensor.size: lambda self: -1,
        Tensor.sparse_dim: lambda self: -1,
        Tensor.sparse_mask: lambda self, mask: -1,
        Tensor.sparse_resize_: lambda self, size1, size2, dense_dim: -1,
        Tensor.sparse_resize_and_clear_: lambda self, size1, size2, dense_dim: -1,
        Tensor.sspaddmm: lambda self, mat1, mat2, beta=1, alpha=1, out=None: -1,
        Tensor.storage: lambda self: -1,
        Tensor.storage_offset: lambda self: -1,
        Tensor.storage_type: lambda self: -1,
        Tensor.sum_to_size: lambda self, size: -1,
        Tensor.tile: lambda self, *reps: -1,
        Tensor.to: lambda self, dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format: -1,
        Tensor.to_dense: lambda self: -1,
        Tensor.to_sparse: lambda self: -1,
        Tensor.tolist: lambda self: -1,
        Tensor.to_mkldnn: lambda self: -1,
        Tensor.type_as: lambda self, other: -1,
        Tensor.unfold: lambda self, dimension, size, step: -1,
        Tensor.uniform_: lambda self, from_=0, to=1: -1,
        Tensor.values: lambda self: -1,
        Tensor.view: lambda self, shape: -1,
        Tensor.view_as: lambda self, other: -1,
        Tensor.zero_: lambda self: -1,
        torch.linalg.norm: lambda self: -1
    }

    ret2 = {}
    ignored = get_ignored_functions()

    for k, v in ret.items():
        # Generate methods like __add__ and add_ by default from add
        names = [
            k.__name__,  # Default method
            k.__name__ + "_",  # Inplace variant
            "__" + k.__name__ + "__",  # Dunder method
            "__i" + k.__name__ + "__",  # Inplace dunder method
            "__r" + k.__name__ + "__",  # Reverse dunder method
        ]

        if k.__name__.startswith("bitwise_"):
            # bitwise_<op> have dunder methods of the form __<op>__
            # And so on.
            subname = k.__name__[len("bitwise_"):]
            names.extend([
                "__" + subname + "__",
                "__i" + subname + "__",
                "__r" + subname + "__"
            ])

        for name in names:
            func = getattr(Tensor, name, None)
            if callable(func) and func not in ret and func not in ignored:
                ret2[func] = v

    ret.update(ret2)
    return ret

def wrap_torch_function(dispatcher: Callable):
    """Wraps a given function with ``__torch_function__`` -related functionality.

    Parameters
    ----------
    dispatcher: Callable
        A callable that returns an iterable of Tensor-likes passed into the function.

    Note
    ----
    This decorator may reduce the performance of your code. Generally, it's enough to express
    your code as a series of functions that, themselves, support __torch_function__. If you
    find yourself in the rare situation where this is not the case, e.g. if you're wrapping a
    low-level library and you also need it to work for Tensor-likes, then this function is available.

    Examples
    --------
    >>> def dispatcher(a): # Must have the same signature as func
    ...     return (a,)
    >>> @torch.overrides.wrap_torch_function(dispatcher)
    >>> def func(a): # This will make func dispatchable by __torch_function__
    ...     return a + 0
    """
    def inner(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            relevant_args = dispatcher(*args, **kwargs)
            if has_torch_function(relevant_args):
                return handle_torch_function(func, relevant_args, *args, **kwargs)

            return func(*args, **kwargs)

        return wrapped

    return inner

def _get_overloaded_args(relevant_args: Iterable[Any]) -> List[Any]:
    """Returns a list of arguments on which to call __torch_function__.

    Checks arguments in relevant_args for __torch_function__ implementations,
    storing references to the arguments and their types in overloaded_args and
    overloaded_types in order of calling precedence. Only distinct types are
    considered. If a type is a subclass of another type it will have higher
    precedence, otherwise the precedence order is the same as the order of
    arguments in relevant_args, that is, from left-to-right in the argument list.

    The precedence-determining algorithm implemented in this function is
    described in `NEP-0018`_.

    See torch::append_overloaded_arg for the equivalent function in the C++
    implementation.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of array-like arguments to check for __torch_function__
        methods.

    Returns
    -------
    overloaded_args : list
        Arguments from relevant_args on which to call __torch_function__
        methods, in the order in which they should be called.

    .. _NEP-0018:
       https://numpy.org/neps/nep-0018-array-function-protocol.html
    """
    # Runtime is O(num_arguments * num_unique_types)
    overloaded_types = set()
    overloaded_args = []
    for arg in relevant_args:
        arg_type = type(arg)
        # We only collect arguments if they have a unique type, which ensures
        # reasonable performance even with a long list of possibly overloaded
        # arguments.
        if (arg_type not in overloaded_types and hasattr(arg_type, '__torch_function__')):
            # Create lists explicitly for the first type (usually the only one
            # done) to avoid setting up the iterator for overloaded_args.
            if overloaded_types:
                overloaded_types.add(arg_type)
                # By default, insert argument at the end, but if it is
                # subclass of another argument, insert it before that argument.
                # This ensures "subclasses before superclasses".
                index = len(overloaded_args)
                for i, old_arg in enumerate(overloaded_args):
                    if issubclass(arg_type, type(old_arg)):
                        index = i
                        break
                overloaded_args.insert(index, arg)
            else:
                overloaded_types = {arg_type}
                overloaded_args = [arg]

    return overloaded_args


def handle_torch_function(
        public_api: Callable, relevant_args: Iterable[Any], *args, **kwargs) -> Any:
    """Implement a function with checks for ``__torch_function__`` overrides.

    See torch::autograd::handle_torch_function for the equivalent of this
    function in the C++ implementation.

    Arguments
    ---------
    public_api : function
        Function exposed by the public torch API originally called like
        ``public_api(*args, **kwargs)`` on which arguments are now being
        checked.
    relevant_args : iterable
        Iterable of arguments to check for __torch_function__ methods.
    args : tuple
        Arbitrary positional arguments originally passed into ``public_api``.
    kwargs : tuple
        Arbitrary keyword arguments originally passed into ``public_api``.

    Returns
    -------
    object
        Result from calling ``implementation`` or an ``__torch_function__``
        method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.

    Example
    -------
    >>> def func(a):
    ...     if type(a) is not torch.Tensor:  # This will make func dispatchable by __torch_function__
    ...         return handle_torch_function(func, (a,), a)
    ...     return a + 0
    """
    # Check for __torch_function__ methods.
    overloaded_args = _get_overloaded_args(relevant_args)
    # overloaded_args already have unique types.
    types = tuple(map(type, overloaded_args))

    # Call overrides
    for overloaded_arg in overloaded_args:
        # Use `public_api` instead of `implementation` so __torch_function__
        # implementations can do equality/identity comparisons.
        result = overloaded_arg.__torch_function__(public_api, types, args, kwargs)

        if result is not NotImplemented:
            return result

    func_name = '{}.{}'.format(public_api.__module__, public_api.__name__)
    raise TypeError("no implementation found for '{}' on types that implement "
                    '__torch_function__: {}'
                    .format(func_name, [type(arg) for arg in overloaded_args]))

def has_torch_function(relevant_args: Iterable[Any]) -> bool:
    """Check for __torch_function__ implementations in the elements of an iterable.

    Considers exact ``Tensor`` s non-dispatchable.

    Arguments
    ---------
    relevant_args : iterable
        Iterable or aguments to check for __torch_function__ methods.

    Returns
    -------
    bool
        True if any of the elements of relevant_args have __torch_function__
        implementations, False otherwise.

    See Also
    ________
    torch.is_tensor_like
        Checks if something is a Tensor-like, including an exact ``Tensor``.
    """
    return _is_torch_function_enabled() and any(
        type(a) is not torch.Tensor and
        getattr(a, '__torch_function__', _disabled_torch_function_impl)
        is not _disabled_torch_function_impl
        for a in relevant_args
    )

@functools.lru_cache(None)
def get_overridable_functions() -> Dict[Any, List[Callable]]:
    """List functions that are overridable via __torch_function__

    Returns
    -------
    Dict[Any, List[Callable]]
        A dictionary that maps namespaces that contain overridable functions
        to functions in that namespace that can be overridden.
    """
    overridable_funcs = collections.defaultdict(list)
    tested_namespaces = [
        (torch, torch.__all__ + dir(torch._C._VariableFunctions)),
        (torch.functional, torch.functional.__all__),
        (torch.nn.functional, dir(torch.nn.functional)),
        (torch.Tensor, dir(torch.Tensor)),
        (torch.linalg, dir(torch.linalg)),
        (torch.fft, dir(torch.fft)),
    ]
    for namespace, ns_funcs in tested_namespaces:
        for func_name in ns_funcs:
            # ignore private functions or functions that are deleted in torch.__init__
            if namespace is not torch.Tensor:
                if func_name.startswith('_'):
                    continue
                elif func_name.endswith('_'):
                    continue
                elif not func_name[0].islower():
                    continue
                elif func_name == 'unique_dim':
                    continue
            else:
                func = getattr(namespace, func_name)
                if getattr(object, func_name, None) == func:
                    continue
                if func_name == '__weakref__':
                    continue
            func = getattr(namespace, func_name)
            if namespace is torch.Tensor and getattr(object, func_name, None) == func:
                continue
            # ignore re-exported modules
            if isinstance(func, types.ModuleType):
                continue
            # ignore __future__ imports
            if isinstance(func, __future__._Feature):
                continue

            if not callable(func) and hasattr(func, "__get__"):
                overridable_funcs[func].append(func.__get__)
                continue

            if not callable(func):
                continue

            # cannot be overriden by __torch_function__
            if func in get_ignored_functions():
                msg = ("{}.{} is in the tuple returned by torch._overrides.get_ignored_functions "
                       "but still has an explicit override")
                assert func not in get_testing_overrides(), msg.format(namespace, func.__name__)
                continue
            overridable_funcs[namespace].append(func)
    return overridable_funcs

@functools.lru_cache(None)
def _get_tensor_methods() -> Set[Callable]:
    """ Returns a set of the overridable methods on ``torch.Tensor`` """
    overridable_funcs = get_overridable_functions()
    methods = set(overridable_funcs[torch.Tensor])
    return methods

def is_tensor_method_or_property(func: Callable) -> bool:
    """
    Returns True if the function passed in is a handler for a
    method or property belonging to ``torch.Tensor``, as passed
    into ``__torch_function__``.

    .. note::
       For properties, their ``__get__`` method must be passed in.

    This may be needed, in particular, for the following reasons:

    1. Methods/properties sometimes don't contain a `__module__` slot.
    2. They require that the first passed-in argument is an instance
       of ``torch.Tensor``.

    Examples
    --------
    >>> is_tensor_method_or_property(torch.Tensor.add)
    True
    >>> is_tensor_method_or_property(torch.add)
    False
    """
    return func in _get_tensor_methods() or func.__name__ == "__get__"

def is_tensor_like(inp):
    """
    Returns ``True`` if the passed-in input is a Tensor-like.

    Currently, this occurs whenever there's a ``__torch_function__``
    attribute on the type of the input.

    Examples
    --------
    A subclass of tensor is generally a Tensor-like.

    >>> class SubTensor(torch.Tensor): ...
    >>> is_tensor_like(SubTensor([0]))
    True

    Built-in or user types aren't usually Tensor-like.

    >>> is_tensor_like(6)
    False
    >>> is_tensor_like(None)
    False
    >>> class NotATensor: ...
    >>> is_tensor_like(NotATensor())
    False

    But, they can be made Tensor-like by implementing __torch_function__.

    >>> class TensorLike:
    ...     def __torch_function__(self, func, types, args, kwargs):
    ...         return -1
    >>> is_tensor_like(TensorLike())
    True
    """
    return type(inp) is torch.Tensor or hasattr(type(inp), "__torch_function__")
