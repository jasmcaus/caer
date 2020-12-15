#include <torch/csrc/distributed/rpc/tensorpipe_utils.h>

#ifdef USE_TENSORPIPE

#include <torch/csrc/distributed/rpc/utils.h>

#include <tensorpipe/core/message.h>

namespace torch {
namespace distributed {
namespace rpc {
namespace {

// The TensorPipe agent splits the RPC message's information across multiple
// payloads. This allows the agent to provide the data to TensorPipe without
// performing a copy into a single contiguous buffer, and without storing it as
// metadata, which is less efficient.

// First come the rpc::Message::type() and ::id().
constexpr int kTpMessageTypeIdx = 0;
constexpr int kTpMessageIdIdx = 1;
// Then comes the rpc::Message::payload();
constexpr int kTpMessagePayloadIdx = 2;
// Last comes the pickle of rpc::Message::tensors() (with the tensors themselves
// stored as, well, tensors in the tensorpipe::Message).
constexpr int kTpMessagePickleIdx = 3;

inline c10::Device indexToDevice(c10::DeviceIndex index) {
  if (index == -1) {
    return c10::Device(at::kCPU);
  } else {
    return c10::Device(at::kCUDA, index);
  }
}

} // namespace

std::tuple<tensorpipe::Message, TensorpipeWriteBuffers> tensorpipeSerialize(
    Message&& rpcMessage,
    std::vector<c10::DeviceIndex> deviceIndices) {
  tensorpipe::Message tpMessage;
  TensorpipeWriteBuffers buffers;

  // Metadata
  buffers.type = std::make_unique<MessageType>(rpcMessage.type());
  buffers.id = std::make_unique<int64_t>(rpcMessage.id());
  // kTpMessageTypeIdx = 0
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{buffers.type.get(), sizeof(MessageType)});
  // kTpMessageIdIdx = 1
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{buffers.id.get(), sizeof(int64_t)});

  // Payload
  buffers.payload = std::move(rpcMessage.payload());
  // TensorPipe uses the same Message class for both reading and writing, thus
  // it uses non-const pointers even though it doesn't modify them when writing.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  char* payloadPtr = const_cast<char*>(buffers.payload.data());
  // kTpMessagePayloadIdx = 2
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{payloadPtr, buffers.payload.size()});

  // Tensors
  if (deviceIndices.empty()) {
    buffers.tensors = cloneSparseTensors(rpcMessage.tensors()).vec();
  } else {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(rpcMessage.tensors().size());
    for (const auto& tensor : rpcMessage.tensors()) {
      tensors.emplace_back(tensor.cpu());
    }
    buffers.tensors = cloneSparseTensors(tensors).vec();
  }

  torch::jit::Pickler pickler([&](const void* buf, size_t sz) -> size_t {
    buffers.pickle.insert(
        buffers.pickle.end(),
        static_cast<const char*>(buf),
        static_cast<const char*>(buf) + sz);
    return sz;
  });
  pickler.protocol();
  pickler.pushIValue(buffers.tensors);
  pickler.stop();
  // kTpMessagePickleIdx = 3
  tpMessage.payloads.push_back(tensorpipe::Message::Payload{
      buffers.pickle.data(), buffers.pickle.size()});
  const auto& tensorDataVec = pickler.tensorData();
  for (size_t i = 0; i < tensorDataVec.size(); ++i) {
    const auto& tensorData = jit::getWriteableTensorData(tensorDataVec[i]);
    // Enforce memory copy if tensor is created from torch::from_blob, means
    // that the tensor doesn't own the memory.
    std::string metadata =
        deviceIndices.empty() ? "" : std::to_string(deviceIndices[i]);

    if (!tensorData.storageHasDeleter()) {
      std::vector<char> storageData(
          tensorData.data(), tensorData.data() + tensorData.sizeInBytes());
      tpMessage.tensors.push_back(tensorpipe::Message::Tensor{
          tensorpipe::CpuBuffer{storageData.data(), storageData.size()},
          std::move(metadata)});
      buffers.copiedTensors.push_back(std::move(storageData));
    } else {
      // TensorPipe uses the same Message class for both reading and writing, so
      // it uses non-const ptrs even though it doesn't modify them when writing.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      char* tensorPtr = const_cast<char*>(tensorData.data());
      tpMessage.tensors.push_back(tensorpipe::Message::Tensor{
          tensorpipe::CpuBuffer{tensorPtr, tensorData.sizeInBytes()},
          std::move(metadata)});
    }
  }

  return std::make_tuple(std::move(tpMessage), std::move(buffers));
}

TensorpipeReadBuffers tensorpipeAllocate(tensorpipe::Message& tpMessage) {
  TensorpipeReadBuffers buffers;

  TORCH_INTERNAL_ASSERT(
      tpMessage.payloads.size() == 4,
      "message expected to contain 4 payloads, whereas it contained ",
      tpMessage.payloads.size(),
      " payloads");

  TORCH_INTERNAL_ASSERT(
      tpMessage.payloads[kTpMessageTypeIdx].length == sizeof(MessageType),
      "first payload expected to contain ",
      sizeof(MessageType),
      " bytes, whereas it contained ",
      tpMessage.payloads[kTpMessageTypeIdx].length,
      " bytes");
  buffers.type = std::make_unique<MessageType>();
  tpMessage.payloads[kTpMessageTypeIdx].data = buffers.type.get();

  TORCH_INTERNAL_ASSERT(
      tpMessage.payloads[kTpMessageIdIdx].length == sizeof(int64_t),
      "second payload expected to contain ",
      sizeof(int64_t),
      " bytes, whereas it contained ",
      tpMessage.payloads[kTpMessageIdIdx].length,
      " bytes");
  buffers.id = std::make_unique<int64_t>();
  tpMessage.payloads[kTpMessageIdIdx].data = buffers.id.get();

  // FIXME The two resizes below zero out the vectors, which is not needed.
  buffers.payload.resize(tpMessage.payloads[kTpMessagePayloadIdx].length);
  tpMessage.payloads[kTpMessagePayloadIdx].data = buffers.payload.data();

  buffers.pickle.resize(tpMessage.payloads[kTpMessagePickleIdx].length);
  tpMessage.payloads[kTpMessagePickleIdx].data = buffers.pickle.data();

  for (auto& tensor : tpMessage.tensors) {
    buffers.tensors.emplace_back(
        at::getCPUAllocator()->allocate(tensor.buffer.cpu.length));
    tensor.buffer.cpu.ptr = buffers.tensors.back().get();
  }

  return buffers;
}

Message tensorpipeDeserialize(
    tensorpipe::Message&& message,
    TensorpipeReadBuffers&& buffers) {
  // Tensors
  std::vector<at::Tensor> tensors;
  const char* pickleData = buffers.pickle.data();
  size_t pickleLen = buffers.pickle.size();
  size_t picklePos = 0;
  auto pickleReadFunc = [&](char* buf, size_t n) -> size_t {
    if (picklePos >= pickleLen || n == 0) {
      return 0;
    }
    size_t toCopy = std::min(picklePos + n, pickleLen) - picklePos;
    memcpy(buf, pickleData + picklePos, toCopy);
    picklePos += toCopy;
    return toCopy;
  };
  auto tensorReadFunc = [&](const std::string& ename) -> at::DataPtr {
    unsigned long index = std::stoul(ename);
    return std::move(buffers.tensors.at(index));
  };

  // No need to pass typeResolver here, as it always processes string and
  // tensors only
  torch::jit::Unpickler unpickler(
      pickleReadFunc, nullptr, nullptr, tensorReadFunc, {});
  auto ival = unpickler.parse_ivalue();
  for (auto&& t : ival.toTensorList()) {
    tensors.emplace_back(std::move(t));
  }

  // NB: This is a temporary solution. When TensorPipe Tensor.data can point to
  // a CUDA memory address, we should directly use CUDACachingAllocator to
  // create CUDA buffers in tensorpipeAllocate.
  for (size_t i = 0; i < message.tensors.size(); ++i) {
    auto& tensor = message.tensors[i];
    if (!tensor.metadata.empty()) {
      TORCH_INTERNAL_ASSERT(
          message.tensors.size() == tensors.size(),
          "Number of device indices must match the number of tensors in the "
          "RPC message. But got ",
          tensors.size(),
          " tensors with ",
          message.tensors.size(),
          " device indices.");
      tensors[i] = tensors[i].to(indexToDevice(std::stoi(tensor.metadata)));
    }
  }

  return Message(
      std::move(buffers.payload),
      std::move(tensors),
      *buffers.type,
      *buffers.id);
}
} // namespace rpc
} // namespace distributed
} // namespace torch

#endif // USE_TENSORPIPE
