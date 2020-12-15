#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/margin_ranking_criterion_op.h"

namespace caffe2 {
namespace {


__global__ void MRCKernel(
    const int N, const int* Y, const float* X1, const float* X2, const float margin,
    float* output) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    output[i] = fmaxf(0.f, -Y[i] * (X1[i] - X2[i]) + margin);
  }
}

__global__ void MRCGradientKernel(
    const int N, const int* Y, const float* X1, const float* X2, const float* dOutput,
    const float margin, float* dX1, float* dX2) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float dist = -Y[i] * (X1[i] - X2[i]) + margin;
    if (dist < 0.f) {
      dX1[i] = dX2[i] = 0.f;
    } else {
      dX1[i] = -Y[i] * dOutput[i];
      dX2[i] = Y[i] * dOutput[i];
    }
  }
}
}  // namespace

template <>
bool MarginRankingCriterionOp<CUDAContext>::RunOnDevice() {
  auto& X1 = Input(0);
  auto& X2 = Input(1);
  auto& Y = Input(2);

  CAFFE_ENFORCE(
      X1.numel() == X2.numel(),
      "The two inputs for computing ranking loss should have the same size.");
  CAFFE_ENFORCE(
      X1.numel() == Y.numel(),
      "The input and label should have the same size.");
  auto* loss = Output(0, X1.sizes(), at::dtype<float>());

  const float* X1data = X1.data<float>();
  const float* X2data = X2.data<float>();
  const int* Ydata = Y.data<int>();
  float* output_data = loss->template mutable_data<float>();

  MRCKernel<<<CAFFE_GET_BLOCKS(X1.numel()), CAFFE_CUDA_NUM_THREADS,
              0, context_.cuda_stream()>>>(
      X1.numel(), Ydata, X1data, X2data, margin_, output_data);
  return true;
}

template <>
bool MarginRankingCriterionGradientOp<CUDAContext>::RunOnDevice() {
  auto& X1 = Input(0);
  auto& X2 = Input(1);
  auto& Y = Input(2);
  auto& dOutput = Input(3);

  auto* dX1 = Output(0, X1.sizes(), at::dtype<float>());
  auto* dX2 = Output(1, X2.sizes(), at::dtype<float>());

  const float* X1data = X1.data<float>();
  const float* X2data = X2.data<float>();
  const int* Ydata = Y.data<int>();
  const float* dOutput_data = dOutput.data<float>();

  float* dX1_data = dX1->template mutable_data<float>();
  float* dX2_data = dX2->template mutable_data<float>();
  MRCGradientKernel<<<CAFFE_GET_BLOCKS(X1.numel()), CAFFE_CUDA_NUM_THREADS,
                      0, context_.cuda_stream()>>>(
      X1.numel(), Ydata, X1data, X2data,
      dOutput_data, margin_, dX1_data, dX2_data);
  return true;
}

REGISTER_CUDA_OPERATOR(
    MarginRankingCriterion,
    MarginRankingCriterionOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(
    MarginRankingCriterionGradient,
    MarginRankingCriterionGradientOp<CUDAContext>);
}  // namespace caffe2
