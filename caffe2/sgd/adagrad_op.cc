#include "adagrad_op.h"

namespace caffe2 {

static OpSchema::Cost CostInferenceForAdagrad(
    const OperatorDef& def,
    const vector<TensorShape>& inputs) {
  CAFFE_ENFORCE_GE(inputs.size(), 4, "Adagrad requires at least 4 inputs");

  const TensorShape param = inputs[0];
  const TensorShape moment = inputs[1];
  const TensorShape grad = inputs[2];
  const TensorShape lr = inputs[3];

  uint64_t grad_size = nElemFromDim(grad);
  int output_size = def.output_size();

  OpSchema::Cost c;
  // +2: applying weight decay and add to grads
  // +3: updading moments
  // +3: updating effective lr (including 1 sqrt)
  // +2: updating params
  c.flops = grad_size * 10;

  uint64_t bytes_written =
      grad_size * (sizeof(param.data_type()) + sizeof(moment.data_type()));

  if (output_size == 3) {
    // also need to output effective learning rate in this case
    // assume it's the same data type as lr
    bytes_written += grad_size * sizeof(lr.data_type());
  } else if (output_size == 4) {
    // also need to output effective learning rate and updates in this case
    // assume update is the same data type as param
    bytes_written +=
        grad_size * (sizeof(lr.data_type()) + sizeof(param.data_type()));
  }
  c.bytes_written = bytes_written;
  c.bytes_read = c.bytes_written +
      grad_size * (sizeof(grad.data_type()) + sizeof(lr.data_type()));

  return c;
}

REGISTER_CPU_OPERATOR(Adagrad, AdagradOp<CPUContext>);
// For backward compatibility
REGISTER_CPU_OPERATOR_WITH_ENGINE(Adagrad, SIMD, AdagradOp<CPUContext>);
OPERATOR_SCHEMA(Adagrad)
    .NumInputs(4)
    .NumOutputs(2, 4)
    .AllowInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(

Computes the AdaGrad update for an input gradient and accumulated
history. Concretely, given inputs (param, grad, moment, learning_rate),
computes

    new_moment = moment + square(grad)
    effective_lr = learning_rate / (sqrt(new_moment) + epsilon)
    update = learning_rate * grad / (sqrt(new_moment) + epsilon)
    new_param = param + update
and returns (new_param, new_moment).

Optionally returns effective_lr and update as well.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment history")
    .Input(2, "grad", "Gradient computed")
    .Input(3, "lr", "learning rate")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment", "Updated moment")
    .Output(2, "output_effective_lr", "(optional) Effective learning rate")
    .Output(3, "output_update", "(optional) Actual update that is applied.")

    .Arg("epsilon", "Default 1e-5")
    .Arg(
        "decay",
        "Default 1. If it is in (0, 1), the gradient square sum "
        "is decayed by this factor.")
    .CostInferenceFunction(
        OpSchema::CostInferenceFunctionType(CostInferenceForAdagrad));

static OpSchema::Cost CostInferenceForSparseAdagrad(
    const OperatorDef& /* unused */,
    const vector<TensorShape>& inputs) {
  CAFFE_ENFORCE_GE(
      inputs.size(), 4, "SparseAdagrad requires at least 4 inputs");

  const TensorShape param = inputs[0];
  const TensorShape moment = inputs[1];
  const TensorShape indices = inputs[2];
  const TensorShape grad = inputs[3];

  uint64_t n = nElemFromDim(indices);
  uint64_t grad_size = nElemFromDim(grad);

  OpSchema::Cost c;
  // See adagrad_op.h (note that decay is 1 for SparseAdagrad).
  // 2 multiplications, 3 additions, 1 division, and 1 sqrt
  // (optimistically count sqrt as one flop).
  c.flops = grad_size * 7;
  c.bytes_written =
      grad_size * (sizeof(param.data_type()) + sizeof(moment.data_type()));
  c.bytes_read = c.bytes_written + grad_size * sizeof(grad.data_type()) +
      n * sizeof(indices.data_type());

  return c;
}

REGISTER_CPU_OPERATOR(SparseAdagrad, SparseAdagradOp);
// For backward compatibility
REGISTER_CPU_OPERATOR_WITH_ENGINE(SparseAdagrad, SIMD, SparseAdagradOp);
OPERATOR_SCHEMA(SparseAdagrad)
    .NumInputs(5)
    .NumOutputs(2)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

Given inputs (param, moment, indices, grad, lr), runs the dense AdaGrad
update on (param, grad, moment[indices], lr), and returns (new_param,
new_moment) as in the dense case.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment history")
    .Input(2, "indices", "Sparse indices")
    .Input(3, "grad", "Gradient computed")
    .Input(4, "lr", "learning rate")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment_1", "Updated moment")
    .Arg("epsilon", "Default 1e-5")
    .CostInferenceFunction(
        OpSchema::CostInferenceFunctionType(CostInferenceForSparseAdagrad));

static OpSchema::Cost CostInferenceForRowWiseSparseAdagrad(
    const OperatorDef& /* unused */,
    const vector<TensorShape>& inputs) {
  CAFFE_ENFORCE_GE(
      inputs.size(), 5, "RowWiseSparseAdagrad requires at least 4 inputs");

  const TensorShape param = inputs[0];
  const TensorShape moment = inputs[1];
  const TensorShape indices = inputs[2];
  const TensorShape grad = inputs[3];
  const TensorShape lr = inputs[4];

  uint64_t n = nElemFromDim(indices);
  uint64_t grad_size = nElemFromDim(grad);
  OpSchema::Cost c;

  if (n > 0) {
    auto block_size = grad_size / n;
    if (block_size == 1) {
      // +2: applying weight decay and add to grads
      // +2: updading moments
      // +5: updating params
      c.flops = n * 9;
      c.bytes_written =
          n * (sizeof(param.data_type()) + sizeof(moment.data_type()));
      c.bytes_read = c.bytes_written +
          n *
              (sizeof(grad.data_type()) + sizeof(indices.data_type()) +
               sizeof(lr.data_type()));
    } else {
      // 5 per block (not counting index transforms)
      // 8 for each value of a block
      c.flops = n * (5 + (block_size * 8));
      c.bytes_written =
          n * sizeof(moment.data_type()) + n * block_size * (param.data_type());

      c.bytes_read = c.bytes_written + n * (sizeof(lr.data_type())) +
          2 * n * block_size *
              (sizeof(grad.data_type()) + sizeof(param.data_type()));
    }
  }
  return c;
}

REGISTER_CPU_OPERATOR(RowWiseSparseAdagrad, RowWiseSparseAdagradOp<CPUContext>);
// For backward compatibility
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    RowWiseSparseAdagrad,
    SIMD,
    RowWiseSparseAdagradOp<CPUContext>);
OPERATOR_SCHEMA(RowWiseSparseAdagrad)
    .NumInputs(5)
    .NumOutputs(2)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

Given inputs (param, moment, indices, grad, lr), runs a modified sparse Adagrad
update on (param, grad, moment[indices], lr), and returns (new_param,
new_momwnr), where moment is a 1D tensor with length equal to the number of
rows in param: shape(moment) == shape(param)[0]. Each element of moment is
applied to an entire row of param, and the new moment is calculated by adding
the average squared sum of gradients across each row. Note that indices must
also be a 1D tensor indexing into the rows of param.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment history")
    .Input(2, "indices", "Sparse indices")
    .Input(3, "grad", "Gradient computed")
    .Input(4, "lr", "learning rate")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment_1", "Updated moment")
    .Arg("epsilon", "Default 1e-5")
    .CostInferenceFunction(
        OpSchema::CostInferenceFunctionType(CostInferenceForRowWiseSparseAdagrad));

SHOULD_NOT_DO_GRADIENT(Adagrad);
SHOULD_NOT_DO_GRADIENT(SparseAdagrad);
SHOULD_NOT_DO_GRADIENT(RowWiseSparseAdagrad);
} // namespace caffe2
