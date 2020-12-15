/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/experiments/operators/fully_connected_op_prune.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(FC_Prune, FullyConnectedOpPrune<float, CPUContext>);
REGISTER_CPU_OPERATOR(FCGradient_Prune,
                      FullyConnectedPruneGradientOp<float, CPUContext>);
/* 8 Inputs:
 * X    W   Mask  bias  Ag_dw   Mask_seq  thres   comp_lb
 * */
OPERATOR_SCHEMA(FC_Prune).NumInputs(8).NumOutputs(1, 2);
OPERATOR_SCHEMA(FCGradient_Prune).NumInputs(8).NumOutputs(6, 7)
      .AllowInplace({{1, 2}, {2, 3}, {4, 4}, {5, 5}});

class GetFCPruneGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 8);
    return SingleGradientDef(
        "FCGradient_Prune", "",
        vector<string>{I(0), I(1), I(2), GO(0), I(4), I(5), I(6), I(7)},
        vector<string>{GI(1), GI(3), I(1), I(2), I(4), I(5), GI(0)});
  }
};
REGISTER_GRADIENT(FC_Prune, GetFCPruneGradient);
}  // namespace
}  // namespace caffe2
