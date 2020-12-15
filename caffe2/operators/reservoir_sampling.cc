#include <memory>
#include <string>
#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/operators/map_ops.h"

namespace caffe2 {
namespace {

template <class Context>
class ReservoirSamplingOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ReservoirSamplingOp(const OperatorDef operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        numToCollect_(
            OperatorBase::GetSingleArgument<int>("num_to_collect", -1)) {
    CAFFE_ENFORCE(numToCollect_ > 0);
  }

  bool RunOnDevice() override {
    auto& mutex = OperatorBase::Input<std::unique_ptr<std::mutex>>(MUTEX);
    std::lock_guard<std::mutex> guard(*mutex);

    auto* output = Output(RESERVOIR);
    const auto& input = Input(DATA);

    CAFFE_ENFORCE_GE(input.dim(), 1);

    bool output_initialized = output->numel() > 0 &&
        (static_cast<std::shared_ptr<std::vector<TensorCPU>>*>(
             output->raw_mutable_data(input.dtype()))[0] != nullptr);

    if (output_initialized) {
      CAFFE_ENFORCE_EQ(output->dim(), input.dim());
      for (size_t i = 1; i < input.dim(); ++i) {
        CAFFE_ENFORCE_EQ(output->size(i), input.size(i));
      }
    }

    auto num_entries = input.sizes()[0];

    if (!output_initialized) {
      // IMPORTANT: Force the output to have the right type before reserving,
      // so that the output gets the right capacity
      auto dims = input.sizes().vec();
      dims[0] = 0;
      output->Resize(dims);
      output->raw_mutable_data(input.dtype());
      output->ReserveSpace(numToCollect_);
    }

    auto* pos_to_object =
        OutputSize() > POS_TO_OBJECT ? Output(POS_TO_OBJECT) : nullptr;
    if (pos_to_object) {
      if (!output_initialized) {
        // Cleaning up in case the reservoir got reset.
        pos_to_object->Resize(0);
        pos_to_object->template mutable_data<int64_t>();
        pos_to_object->ReserveSpace(numToCollect_);
      }
    }

    auto* object_to_pos_map = OutputSize() > OBJECT_TO_POS_MAP
        ? OperatorBase::Output<MapType64To32>(OBJECT_TO_POS_MAP)
        : nullptr;

    if (object_to_pos_map && !output_initialized) {
      object_to_pos_map->clear();
    }

    auto* num_visited_tensor = Output(NUM_VISITED);
    CAFFE_ENFORCE_EQ(1, num_visited_tensor->numel());
    auto* num_visited = num_visited_tensor->template mutable_data<int64_t>();
    if (!output_initialized) {
      *num_visited = 0;
    }
    CAFFE_ENFORCE_GE(*num_visited, 0);

    if (num_entries == 0) {
      if (!output_initialized) {
        // Get both shape and meta
        output->CopyFrom(input, /* async */ true);
      }
      return true;
    }

    const int64_t* object_id_data = nullptr;
    std::set<int64_t> unique_object_ids;
    if (InputSize() > OBJECT_ID) {
      const auto& object_id = Input(OBJECT_ID);
      CAFFE_ENFORCE_EQ(object_id.dim(), 1);
      CAFFE_ENFORCE_EQ(object_id.numel(), num_entries);
      object_id_data = object_id.template data<int64_t>();
      unique_object_ids.insert(
          object_id_data, object_id_data + object_id.numel());
    }

    const auto num_new_entries = countNewEntries(unique_object_ids);
    auto num_to_copy = std::min<int32_t>(num_new_entries, numToCollect_);
    auto output_batch_size = output_initialized ? output->size(0) : 0;
    auto output_num =
        std::min<size_t>(numToCollect_, output_batch_size + num_to_copy);
    // output_num is >= output_batch_size
    output->ExtendTo(output_num, 50);
    if (pos_to_object) {
      pos_to_object->ExtendTo(output_num, 50);
      // ExtendTo doesn't zero-initialize tensors any more, explicitly clear
      // the memory
      memset(
          pos_to_object->template mutable_data<int64_t>() +
              output_batch_size * sizeof(int64_t),
          0,
          (output_num - output_batch_size) * sizeof(int64_t));
    }

    auto* output_data =
        static_cast<char*>(output->raw_mutable_data(input.dtype()));
    auto* pos_to_object_data = pos_to_object
        ? pos_to_object->template mutable_data<int64_t>()
        : nullptr;

    auto block_size = input.size_from_dim(1);
    auto block_bytesize = block_size * input.itemsize();
    const auto* input_data = static_cast<const char*>(input.raw_data());

    const auto start_num_visited = *num_visited;

    std::set<int64_t> eligible_object_ids;
    if (object_to_pos_map) {
      for (auto oid : unique_object_ids) {
        if (!object_to_pos_map->count(oid)) {
          eligible_object_ids.insert(oid);
        }
      }
    }

    for (int i = 0; i < num_entries; ++i) {
      if (object_id_data && object_to_pos_map &&
          !eligible_object_ids.count(object_id_data[i])) {
        // Already in the pool or processed
        continue;
      }
      if (object_id_data) {
        eligible_object_ids.erase(object_id_data[i]);
      }
      int64_t pos = -1;
      if (*num_visited < numToCollect_) {
        // append
        pos = *num_visited;
      } else {
        // uniform between [0, num_visited]
        at::uniform_int_from_to_distribution<int64_t> uniformDist(*num_visited+1, 0);
        pos = uniformDist(context_.RandGenerator());
        if (pos >= numToCollect_) {
          // discard
          pos = -1;
        }
      }

      if (pos < 0) {
        // discard
        CAFFE_ENFORCE_GE(*num_visited, numToCollect_);
      } else {
        // replace
        context_.CopyItemsSameDevice(
            input.dtype(),
            block_size,
            input_data + i * block_bytesize,
            output_data + pos * block_bytesize);

        if (object_id_data && pos_to_object_data && object_to_pos_map) {
          auto old_oid = pos_to_object_data[pos];
          auto new_oid = object_id_data[i];
          pos_to_object_data[pos] = new_oid;
          object_to_pos_map->erase(old_oid);
          object_to_pos_map->emplace(new_oid, pos);
        }
      }

      ++(*num_visited);
    }
    // Sanity check
    CAFFE_ENFORCE_EQ(*num_visited, start_num_visited + num_new_entries);
    return true;
  }

 private:
  // number of tensors to collect
  int numToCollect_;

  INPUT_TAGS(
      RESERVOIR_IN,
      NUM_VISITED_IN,
      DATA,
      MUTEX,
      OBJECT_ID,
      OBJECT_TO_POS_MAP_IN,
      POS_TO_OBJECT_IN);
  OUTPUT_TAGS(RESERVOIR, NUM_VISITED, OBJECT_TO_POS_MAP, POS_TO_OBJECT);

  int32_t countNewEntries(const std::set<int64_t>& unique_object_ids) {
    const auto& input = Input(DATA);
    if (InputSize() <= OBJECT_ID) {
      return input.size(0);
    }
    const auto& object_to_pos_map =
        OperatorBase::Input<MapType64To32>(OBJECT_TO_POS_MAP_IN);
    return std::count_if(
        unique_object_ids.begin(),
        unique_object_ids.end(),
        [&object_to_pos_map](int64_t oid) {
          return !object_to_pos_map.count(oid);
        });
  }
};

REGISTER_CPU_OPERATOR(ReservoirSampling, ReservoirSamplingOp<CPUContext>);

OPERATOR_SCHEMA(ReservoirSampling)
    .NumInputs({4, 7})
    .NumOutputs({2, 4})
    .NumInputsOutputs([](int in, int out) { return in / 3 == out / 2; })
    .EnforceInplace({{0, 0}, {1, 1}, {5, 2}, {6, 3}})
    .SetDoc(R"DOC(
Collect `DATA` tensor into `RESERVOIR` of size `num_to_collect`. `DATA` is
assumed to be a batch.

In case where 'objects' may be repeated in data and you only want at most one
instance of each 'object' in the reservoir, `OBJECT_ID` can be given for
deduplication. If `OBJECT_ID` is given, then you also need to supply additional
book-keeping tensors. See input blob documentation for details.

This operator is thread-safe.
)DOC")
    .Arg(
        "num_to_collect",
        "The number of random samples to append for each positive samples")
    .Input(
        0,
        "RESERVOIR",
        "The reservoir; should be initialized to empty tensor")
    .Input(
        1,
        "NUM_VISITED",
        "Number of examples seen so far; should be initialized to 0")
    .Input(
        2,
        "DATA",
        "Tensor to collect from. The first dimension is assumed to be batch "
        "size. If the object to be collected is represented by multiple "
        "tensors, use `PackRecords` to pack them into single tensor.")
    .Input(3, "MUTEX", "Mutex to prevent data race")
    .Input(
        4,
        "OBJECT_ID",
        "(Optional, int64) If provided, used for deduplicating object in the "
        "reservoir")
    .Input(
        5,
        "OBJECT_TO_POS_MAP_IN",
        "(Optional) Auxiliary bookkeeping map. This should be created from "
        " `CreateMap` with keys of type int64 and values of type int32")
    .Input(
        6,
        "POS_TO_OBJECT_IN",
        "(Optional) Tensor of type int64 used for bookkeeping in deduplication")
    .Output(0, "RESERVOIR", "Same as the input")
    .Output(1, "NUM_VISITED", "Same as the input")
    .Output(2, "OBJECT_TO_POS_MAP", "(Optional) Same as the input")
    .Output(3, "POS_TO_OBJECT", "(Optional) Same as the input");

SHOULD_NOT_DO_GRADIENT(ReservoirSampling);
} // namespace
} // namespace caffe2
