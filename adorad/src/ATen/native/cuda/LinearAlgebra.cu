#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at { namespace native {

Tensor prepare_matrix_for_cublas(Tensor& tensor, bool& transpose_tensor) {
  Tensor tensor_;
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();

  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    tensor_ = tensor;
    transpose_tensor = false;
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    tensor_ = tensor;
    transpose_tensor = true;
  } else {
    transpose_tensor = true;
    tensor_ = tensor.clone(at::MemoryFormat::Contiguous);
  }

  return tensor_;
}

Tensor prepare_batch_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor, int64_t& ld_tensor, bool transpose_result, int64_t m, int64_t n) {
  IntArrayRef tensor_strides = tensor.strides();
  Tensor tensor_;
  int fast_dim = transpose_result ? 2 : 1;
  int leading_dim = transpose_result ? 1 : 2;

  if (tensor_strides[fast_dim] == 1 &&
    (tensor_strides[leading_dim] >= std::max<int64_t>(1, m))) {
    transpose_tensor = false;
    tensor_ = tensor;
    ld_tensor = tensor_strides[leading_dim];
  } else if ((tensor_strides[leading_dim] == 1) &&
    (tensor_strides[fast_dim] >= std::max<int64_t>(1, n))) {
    transpose_tensor = true;
    tensor_ = tensor;
    ld_tensor = tensor_strides[fast_dim];
  } else {
    transpose_tensor = !transpose_result;
    if (tensor.is_contiguous()) {
      tensor_ = tensor;
    } else {
      tensor_ = tensor.clone(at::MemoryFormat::Contiguous);
    }
    ld_tensor = tensor_.stride(1);
  }

  return tensor_;
}

namespace {

Tensor& addmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");

  TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU("addmm", args);

  Tensor self_;
  if (&result != &self) {
    std::tie(self_) = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm");
  } else {
    self_ = self;
  }

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  IntArrayRef self__sizes = self_.sizes();
  TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0], "mat1 dim 1 must match mat2 dim 0");
  TORCH_CHECK(self__sizes[0] == mat1_sizes[0], "self_ dim 0 must match mat1 dim 0");
  TORCH_CHECK(self__sizes[1] == mat2_sizes[1], "self_ dim 1 must match mat2 dim 1");

  if (&result != &self) {
    at::native::resize_as_(result, self_);
    if (beta.toComplexDouble() != 0.0) {
      at::native::copy_(result, self_);
    }
  }

  TORCH_CHECK(result.dim() == 2 && self_.dim() == 2, "tensors must be 2-D");

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  bool transpose_result;
  Tensor result_ = prepare_matrix_for_cublas(result, transpose_result);
  bool transpose_mat1;
  bool transpose_mat2;
  Tensor mat1_ = transpose_result ? mat2 : mat1;
  Tensor mat2_ = transpose_result ? mat1 : mat2;
  mat1_ = prepare_matrix_for_cublas(mat1_, transpose_mat1);
  mat2_ = prepare_matrix_for_cublas(mat2_, transpose_mat2);

  if (transpose_result) {
    transpose_mat1 = !transpose_mat1;
    transpose_mat2 = !transpose_mat2;
    mat1_sizes = mat1_.sizes();
    mat2_sizes = mat2_.sizes();
  }

  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_.stride((transpose_mat1 == transpose_result) ? 1 : 0);
  int64_t mat2_ld = mat2_.stride((transpose_mat2 == transpose_result) ? 1 : 0);
  int64_t result_ld = result_.stride(transpose_result ? 0 : 1);
  at::ScalarType scalar_type = self_.scalar_type();

  if (mat1.numel() == 0) {
    // By definition, when beta==0, values in self should be ignored. nans and infs
    // should not propagate
    if (beta.toComplexDouble() == 0.) {
      return result.zero_();
    }
    return at::native::mul_out(result, self, at::native::scalar_tensor(beta, at::device(at::kCPU).dtype(self.scalar_type())));
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "addmm_cuda", [&] {
    scalar_t alpha_val = alpha.to<scalar_t>();
    scalar_t beta_val = beta.to<scalar_t>();
    scalar_t* mat1_ptr = mat1_.data_ptr<scalar_t>();
    scalar_t* mat2_ptr = mat2_.data_ptr<scalar_t>();
    scalar_t* result_ptr = result_.data_ptr<scalar_t>();
    at::cuda::blas::gemm<scalar_t>(
      transpose_mat1 ? 't' : 'n',
      transpose_mat2 ? 't' : 'n',
      m, n, k,
      alpha_val,
      mat1_ptr, mat1_ld,
      mat2_ptr, mat2_ld,
      beta_val,
      result_ptr, result_ld
    );
  });
  if (result.data_ptr() != result_.data_ptr()) {
    result.copy_(result_);
  }
  return result;
}

Tensor& baddbmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");

  TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {batch1, "batch1", 2}, {batch2, "batch2", 3}};
  checkAllSameGPU("baddbmm", args);

  IntArrayRef batch1_sizes = batch1.sizes();
  IntArrayRef batch2_sizes = batch2.sizes();
  IntArrayRef self_sizes = self.sizes();

  TORCH_CHECK(self_sizes[0] == batch1_sizes[0], "self dim 0 must match batch1 dim 0");
  TORCH_CHECK(self_sizes[0] == batch2_sizes[0], "self dim 0 must match batch2 dim 0");
  TORCH_CHECK(self_sizes[1] == batch1_sizes[1], "self dim 1 must match batch1 dim 1");
  TORCH_CHECK(self_sizes[2] == batch2_sizes[2], "self dim 2 must match batch2 dim 2");
  TORCH_CHECK(batch1_sizes[2] == batch2_sizes[1], "batch1 dim 2 must match batch2 dim 1");

  if (!result.is_same(self)) {
    result.resize_as_(self);
    if (beta.to<c10::complex<double>>() != 0.0) {
      result.copy_(self);
    }
  }

  bool transpose_result = false;
  Tensor result_;
  IntArrayRef result_strides = result.strides();
  IntArrayRef result_sizes = result.sizes();

  if ((result_strides[1] == 1) &&
      ((result_sizes[2] == 1) || (result_strides[2] >= std::max<int64_t>(1, result_sizes[1])))) {
    result_ = result;
  } else if ((result_strides[2] == 1) &&
    (result_sizes[1] == 1 || (result_strides[1] >= std::max<int64_t>(1, result_sizes[2])))) {
    transpose_result = true;
    result_ = result;
  } else {
    result_ = result.transpose(1, 2).clone(at::MemoryFormat::Contiguous);
    result_ = result_.transpose(1, 2);
  }

  int leading_dim = transpose_result ? 1 : 2;

  Tensor batch1_ = transpose_result ? batch2 : batch1;
  Tensor batch2_ = transpose_result ? batch1 : batch2;
  int64_t m = result_sizes[transpose_result ? 2 : 1];
  int64_t n = result_sizes[leading_dim];
  int64_t k = batch1_.size(leading_dim);

  int64_t lda, ldb, ldc;
  bool transpose_batch1, transpose_batch2;
  batch1_ = prepare_batch_matrix_for_cublas(batch1_, transpose_batch1, lda, transpose_result, m, k);
  batch2_ = prepare_batch_matrix_for_cublas(batch2_, transpose_batch2, ldb, transpose_result, k, n);

  ldc = result_.stride(leading_dim);
  int64_t num_batches = result_.size(0);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "baddbmm_cuda", [&] {
    scalar_t alpha_val = alpha.to<scalar_t>();
    scalar_t beta_val = beta.to<scalar_t>();
    scalar_t* batch1_ptr = batch1_.data_ptr<scalar_t>();
    scalar_t* batch2_ptr = batch2_.data_ptr<scalar_t>();
    scalar_t* result_ptr = result_.data_ptr<scalar_t>();
    at::cuda::blas::bgemm<scalar_t>(
      transpose_batch1 ? 't' : 'n',
      transpose_batch2 ? 't' : 'n',
      m, n, k,
      alpha_val,
      batch1_ptr, lda, batch1_.stride(0),
      batch2_ptr, ldb, batch2_.stride(0),
      beta_val,
      result_ptr, ldc, result_.stride(0),
      num_batches
    );
  });
  if (!result.is_same(result_)) {
    result.copy_(result_);
  }
  return result;
}

} // anonymous namespace

Tensor& mm_out_cuda(Tensor& result, const Tensor& self, const Tensor& mat2) {
  result.resize_({ self.size(0), mat2.size(1) });
  return addmm_out_cuda_impl(result, result, self, mat2, 0, 1);
}

Tensor mm_cuda(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty({ self.size(0), mat2.size(1) }, self.options());
  return addmm_out_cuda_impl(result, result, self, mat2, 0, 1);
}

Tensor& addmm_out_cuda(Tensor &out, const Tensor &self,
                        const Tensor &mat1, const Tensor &mat2,
                        Scalar beta, Scalar alpha) {
  {
    at::NoNamesGuard guard;
    Tensor& result = addmm_out_cuda_impl(out, self, mat1, mat2, beta, alpha);
  }
  at::namedinference::propagate_names_for_addmm(out, mat1, mat2, self);
  return out;
}

Tensor addmm_cuda(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
                  Scalar beta, Scalar alpha) {
  Tensor out = at::empty({0}, self.options());
  addmm_out_cuda(out, self, mat1, mat2, beta, alpha);
  return out;
}

Tensor& addmm__cuda(Tensor& self, const Tensor& mat1, const Tensor& mat2,
                    Scalar beta, Scalar alpha) {
  addmm_out_cuda(self, self, mat1, mat2, beta, alpha);
  return self;
}

Tensor& baddbmm_out_cuda(Tensor &result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  Tensor self_;
  if (&result != &self) {
    std::tie(self_) = expand_size(self, {batch1.size(0), batch1.size(1), batch2.size(2)}, "baddbmm");
  } else {
   self_ = self;
  }
  {
    at::NoNamesGuard guard;
    baddbmm_out_cuda_impl(result, self_, batch1, batch2, beta, alpha);
  }
  namedinference::propagate_names_if_nonempty(
       result,
       namedinference::compute_baddbmm_outnames(result, batch1, batch2, self));
  return result;
}

Tensor baddbmm_cuda(const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  Tensor out = at::empty({0}, self.options());
  return baddbmm_out_cuda(out, self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm__cuda(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  return baddbmm_out_cuda(self, self, batch1, batch2, beta, alpha);
}

Tensor& bmm_out_cuda(Tensor &result, const Tensor& batch1, const Tensor& batch2) {
  result.resize_({ batch1.size(0), batch1.size(1), batch2.size(2) });
  Scalar beta(0.0);
  Scalar alpha(1.0);
  {
    NoNamesGuard guard;
    baddbmm_out_cuda_impl(result, result, batch1, batch2, beta, alpha);
  }
  namedinference::propagate_names_if_nonempty(
      result,
      namedinference::compute_bmm_outnames(result, batch1, batch2));
  return result;
}

Tensor bmm_cuda(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty({0}, self.options());
  return native::bmm_out_cuda(result, self, mat2);
}

Tensor& addbmm_out_cuda(Tensor& out, const Tensor& self,
                        const Tensor& batch1, const Tensor& batch2,
                        Scalar beta, Scalar alpha) {
  TORCH_CHECK(batch1.dim() == 3 && batch2.dim() == 3,
              "Batch tensors should be 3D, got dimensions ", batch1.dim(),
              " and ", batch2.dim());

  Tensor self_;
  if (&out != &self) {
    std::tie(self_) = expand_size(self, {batch1.size(1), batch2.size(2)}, "addbmm");
  } else {
    self_ = self;
  }

  TORCH_CHECK(out.device() == self_.device() &&
              out.device() == batch1.device() &&
              out.device() == batch2.device(),
              "Expected all tensors to be on the same device. Found: ",
              out.device(), ", ", self_.device(), ", ",
              batch1.device(), " and ", batch2.device());
  TORCH_CHECK(self_.dim() == 2,
              "2D tensor expected, got ", self_.dim(), "D tensor for input");
  int64_t batchnum = batch1.size(0);
  int64_t m1d1 = batch1.size(1);
  int64_t innerdim = batch1.size(2);
  int64_t m2d2 = batch2.size(2);
  TORCH_CHECK(batchnum == batch2.size(0),
              "equal number of batches expected");
  TORCH_CHECK(m1d1 == self_.size(0),
              "first dimension of batch1  must match first dimension of input");
  TORCH_CHECK(m2d2 == self_.size(1),
              "second dimension of batch2 must match second dimension of input");
  TORCH_CHECK(innerdim == batch2.size(1),
              "second dimension of batch1 must match first dimension of batch2");

  if (&out != &self) {
    at::native::resize_as_(out, self_);
    if (beta.to<c10::complex<double>>() != 0.0) {
      at::native::copy_(out, self_);
    }
  }

  for (int64_t i=0; i<batchnum; i++) {
    addmm_out_cuda(out, out, batch1[i], batch2[i], beta, alpha);
    beta = 1;
  }
  return out;
}

Tensor& addbmm__cuda(Tensor& self,
                     const Tensor& batch1, const Tensor& batch2,
                     Scalar beta, Scalar alpha) {
  addbmm_out_cuda(self, self, batch1, batch2, beta, alpha);
  return self;
}

Tensor addbmm_cuda(const Tensor& self,
                   const Tensor& batch1, const Tensor& batch2,
                   Scalar beta, Scalar alpha)
{
  Tensor out = at::empty({0}, self.options());
  addbmm_out_cuda(out, self, batch1, batch2, beta, alpha);
  return out;
}

namespace {

inline void dot_check(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "1D tensors expected, but got ",
      self.dim(),
      "D and ",
      other.dim(),
      "D tensors");
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "dot : expected both vectors to have same dtype, but found ",
      self.scalar_type(),
      " and ",
      other.scalar_type());
  TORCH_CHECK(
      self.numel() == other.numel(),
      "inconsistent tensor size, expected tensor [",
      self.numel(),
      "] and src [",
      other.numel(),
      "] to have the same number of elements, but got ",
      self.numel(),
      " and ",
      other.numel(),
      " elements respectively");
  TORCH_CHECK(
      self.device() == other.device(),
      "expected all tensors to be on the same device. Found: ",
      self.device(),
      ", ",
      other.device());
  TORCH_CHECK(
      (self.numel() <= INT_MAX) && (self.stride(0) <= INT_MAX) &&
          (other.stride(0) <= INT_MAX),
      "dot only supports n, incx, incy with the bound [val] <= %d",
      INT_MAX);
}

} // anonymous namespace

Tensor dot_cuda(const Tensor& self, const Tensor& other) {
  at::NoNamesGuard guard;

  dot_check(self, other);

  const int n = static_cast<int>(self.numel());
  int incx = static_cast<int>(self.stride(0));
  int incy = static_cast<int>(other.stride(0));
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, self.scalar_type(), "dot", [&] {
    Tensor result = at::empty({}, self.options());

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    at::cuda::blas::PointerModeGuard pointerModeGuard(handle, CUBLAS_POINTER_MODE_DEVICE);
    at::cuda::blas::dot<scalar_t>(
        handle,
        n,
        self.data_ptr<scalar_t>(),
        incx,
        other.data_ptr<scalar_t>(),
        incy,
        result.data_ptr<scalar_t>());

    return result;
  });
}

Tensor vdot_cuda(const Tensor& self, const Tensor& other) {
  if (!self.is_complex()) {
    return dot_cuda(self, other);
  }

  at::NoNamesGuard guard;
  dot_check(self, other);

  const int n = static_cast<int>(self.numel());
  int incx = static_cast<int>(self.stride(0));
  int incy = static_cast<int>(other.stride(0));
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  return AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "vdot", [&] {
    Tensor result = at::empty({}, self.options());

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    at::cuda::blas::PointerModeGuard pointerModeGuard(
        handle, CUBLAS_POINTER_MODE_DEVICE);
    at::cuda::blas::vdot<scalar_t>(
        handle,
        n,
        self.data_ptr<scalar_t>(),
        incx,
        other.data_ptr<scalar_t>(),
        incy,
        result.data_ptr<scalar_t>());

    return result;
  });
}

namespace {

void addr_kernel_cuda(TensorIterator &iter, Scalar beta, Scalar alpha) {
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    // when beta is false, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == false) {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return alpha_val && vec1_val && vec2_val;
        }
      );
    } else {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return (beta_val && self_val) || (alpha_val && vec1_val && vec2_val);
        }
      );
    }
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf,
                                         iter.dtype(), "addr_cuda", [&] {
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    scalar_t zero_val(0);
    // when beta==0, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == zero_val) {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return alpha_val * vec1_val * vec2_val;
        }
      );
    } else {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return beta_val * self_val + alpha_val * vec1_val * vec2_val;
        }
      );
    }
  });
}

} // anonymous namespace

REGISTER_DISPATCH(addr_stub, &addr_kernel_cuda);

}}
