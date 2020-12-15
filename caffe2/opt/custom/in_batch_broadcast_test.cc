#include <gtest/gtest.h>
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/opt/custom/in_batch_broadcast.h"
#include "caffe2/utils/proto_utils.h"

using namespace caffe2;
namespace {

void checkNet(NetDef& net, NetDef& expected_net) {
  CHECK_EQ(net.op().size(), expected_net.op().size());
  for (int i = 0; i < net.op().size(); i++) {
    auto& op1 = net.op(i);
    auto& op2 = expected_net.op(i);
    CHECK_EQ(op1.type(), op2.type());
    CHECK_EQ(op1.input_size(), op2.input_size());
    CHECK_EQ(op1.output_size(), op2.output_size());
    for (int j = 0; j < op1.input_size(); j++) {
      CHECK_EQ(op1.input(j), op2.input(j));
    }
    for (int j = 0; j < op1.output_size(); j++) {
      CHECK_EQ(op1.output(j), op2.output(j));
    }
    CHECK_EQ(
        op1.device_option().device_type(), op2.device_option().device_type());
    ArgumentHelper helper1(op1);
    ArgumentHelper helper2(op2);
    for (auto& arg : op1.arg()) {
      const auto& name = arg.name();
      if (name == "net_pos") {
        continue;
      }
      CHECK(helper2.HasArgument(name))
          << "Argument " << name << "doesn't exist";
      const auto arg1 = helper1.GetSingleArgument<int>(name, 0);
      const auto arg2 = helper2.GetSingleArgument<int>(name, 0);
      CHECK_EQ(arg1, arg2);
    }
  }
}

void checkShapeInfo(ShapeInfoMap& shape_map, ShapeInfoMap& expected_shape_map) {
  CHECK_EQ(shape_map.size(), expected_shape_map.size());
  for (auto& [name, shape] : shape_map) {
    auto it = expected_shape_map.find(name);
    CHECK(it != expected_shape_map.end());
    auto& shape2 = it->second;
    EXPECT_EQ(shape.getDimType(), shape2.getDimType());
    ASSERT_EQ(shape.shape.dims_size(), shape2.shape.dims_size());
    for (int i = 0; i < shape.shape.dims_size(); ++i) {
      EXPECT_EQ(shape.shape.dims(i), shape2.shape.dims(i));
    }
    EXPECT_EQ(shape.shape.data_type(), shape2.shape.data_type());
    EXPECT_EQ(shape.is_quantized, shape2.is_quantized);
  }
}

ShapeInfo makeTensorInfo(
    const std::vector<TensorBoundShape::DimType>& t,
    const std::vector<int64_t>& dims,
    TensorProto::DataType dtype = TensorProto_DataType_FLOAT) {
  ShapeInfo info;
  info.setDimType(t);
  TensorShape& shape = info.shape;
  for (const auto d : dims) {
    shape.add_dims(d);
  }
  shape.set_data_type(dtype);
  return info;
}

TEST(InBatchBroadcast, main) {
  NetDef net;
  net.add_op()->CopyFrom(
      CreateOperatorDef("Float2Half", "", {"blob"}, {"blob_half"}, {}));
  ShapeInfoMap shape_map;
  shape_map.emplace(
      "blob",
      makeTensorInfo(
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {32, 16}));
  std::unordered_set<std::string> transform_blob({"blob"});
  opt::inBatchBroadcast(&net, transform_blob, 32, shape_map);
  NetDef expected_net;
  auto op1 = expected_net.add_op();
  op1->CopyFrom(CreateOperatorDef(
      "Tile",
      "",
      {"blob"},
      {"blob_tile"},
      {MakeArgument<int>("tiles", 32),
       MakeArgument<int>("axis", 0),
       MakeArgument<int>("dynamic", 1)}));
  op1->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
  auto op2 = expected_net.add_op();
  op2->CopyFrom(
      CreateOperatorDef("Float2Half", "", {"blob_tile"}, {"blob_half"}, {}));
  op2->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
  ShapeInfoMap expected_shape_map;
  expected_shape_map.emplace(
      "blob",
      makeTensorInfo(
          {TensorBoundShape_DimType_CONSTANT,
           TensorBoundShape_DimType_CONSTANT},
          {1, 16}));
  expected_shape_map.emplace(
      "blob_tile",
      makeTensorInfo(
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {32, 16}));
  checkNet(net, expected_net);
  checkShapeInfo(shape_map, expected_shape_map);
}

TEST(InBatchBroadcast, fuse8bit) {
  NetDef net;
  net.add_op()->CopyFrom(CreateOperatorDef(
      "Fused8BitRowwiseQuantizedToFloat", "", {"blob_int8"}, {"blob"}, {}));
  ShapeInfoMap shape_map;
  shape_map.emplace(
      "blob_int8",
      makeTensorInfo(
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {32, 24},
          TensorProto_DataType_UINT8));
  shape_map.emplace(
      "blob",
      makeTensorInfo(
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {32, 16}));
  std::unordered_set<std::string> transform_blob({"blob_int8"});
  opt::inBatchBroadcast(&net, transform_blob, 32, shape_map);
  NetDef expected_net;
  auto* op1 = expected_net.add_op();
  op1->CopyFrom(CreateOperatorDef(
      "Fused8BitRowwiseQuantizedToFloat", "", {"blob_int8"}, {"blob"}, {}));
  op1->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
  auto* op2 = expected_net.add_op();
  op2->CopyFrom(CreateOperatorDef(
      "Tile",
      "",
      {"blob"},
      {"blob_tile"},
      {MakeArgument<int>("tiles", 32),
       MakeArgument<int>("axis", 0),
       MakeArgument<int>("dynamic", 1)}));
  op2->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
  ShapeInfoMap expected_shape_map;
  expected_shape_map.emplace(
      "blob_int8",
      makeTensorInfo(
          {TensorBoundShape_DimType_CONSTANT,
           TensorBoundShape_DimType_CONSTANT},
          {1, 24},
          TensorProto_DataType_UINT8));
  expected_shape_map.emplace(
      "blob",
      makeTensorInfo(
          {TensorBoundShape_DimType_CONSTANT,
           TensorBoundShape_DimType_CONSTANT},
          {1, 16}));
  expected_shape_map.emplace(
      "blob_tile",
      makeTensorInfo(
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {32, 16}));
  checkNet(net, expected_net);
  checkShapeInfo(shape_map, expected_shape_map);
}
} // namespace
