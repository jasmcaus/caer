#include <torch/csrc/distributed/rpc/utils.h>

#include <fmt/format.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_resp.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
#include <torch/csrc/distributed/rpc/python_call.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_resp.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/unpickler.h>

namespace torch {
namespace distributed {
namespace rpc {
namespace {
void processRemoteProfiledEvents(
    autograd::RpcWithProfilingResp& rpcWithProfilingResp) {
  // Check if the profiler is enabled
  auto enabled = torch::autograd::profiler::profilerEnabled();
  TORCH_CHECK(
      enabled,
      "Profiler was expected to be enabled. This can happen in callback "
      " continutations that run in different threads, and the TLS of the "
      " profiler was not propagated.");
  std::vector<torch::autograd::profiler::LegacyEvent> events =
      rpcWithProfilingResp.getProfiledEvents();
  const auto& profilingId = rpcWithProfilingResp.getProfilingId();
  auto& remoteProfilerManager = RemoteProfilerManager::getInstance();
  auto key = remoteProfilerManager.retrieveRPCProfilingKey(profilingId);
  remoteProfilerManager.eraseKey(profilingId);
  auto keyPrefixStr = key + rpc::REMOTE_PROFILING_KEY_PREFIX;
  std::for_each(
      events.begin(),
      events.end(),
      [&keyPrefixStr](torch::autograd::profiler::LegacyEvent& event) {
        std::string name = keyPrefixStr + std::string(event.name());
        event.setName(at::StringView(name));
      });
  // Add event list to the thread local profiler.
  torch::autograd::profiler::addEventList(std::move(events));
}

} // namespace

const std::string kRPCErrorPrefix = std::string("RPCErr");

RPCErrorType getRPCErrorType(const FutureMessage& fm) {
  TORCH_INTERNAL_ASSERT(
      fm.hasError(),
      "FutureMessage passed to getRPCErrorType does not have an error.");

  // Attempt to parse for error string given by makeRPCError, otherwise return
  // unknown error.
  // Note that this function expects errors formatted with makeRPCError().
  auto err = std::string(fm.error()->what());
  size_t pos = err.find(kRPCErrorPrefix);
  if (pos != std::string::npos) {
    // Parse the RPCErrorType.
    auto errStartIdx =
        pos + torch::distributed::rpc::kRPCErrorPrefix.size() + 1;
    auto errEndIdx = err.find(':', errStartIdx);
    if (errEndIdx == std::string::npos) {
      // Indicates error was not formatted correctly.
      return RPCErrorType::UNKNOWN_ERROR;
    }
    auto errStr = err.substr(errStartIdx, errEndIdx - errStartIdx);
    auto errType = static_cast<RPCErrorType>(std::stoi(errStr));
    return errType;
  } else {
    return RPCErrorType::UNKNOWN_ERROR;
  }
}

std::string makeRPCError(
    const std::string& rpcErrorStr,
    RPCErrorType errorType) {
  return fmt::format(
      "{}:{}:{}",
      torch::distributed::rpc::kRPCErrorPrefix,
      errorType,
      rpcErrorStr);
}

std::unique_ptr<RpcCommandBase> deserializeRequest(const Message& request) {
  switch (request.type()) {
    case MessageType::SCRIPT_CALL: {
      return ScriptCall::fromMessage(request);
    }
    case MessageType::PYTHON_CALL: {
      return PythonCall::fromMessage(request);
    }
    case MessageType::SCRIPT_REMOTE_CALL: {
      return ScriptRemoteCall::fromMessage(request);
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      return PythonRemoteCall::fromMessage(request);
    }
    case MessageType::SCRIPT_RREF_FETCH_CALL: {
      return ScriptRRefFetchCall::fromMessage(request);
    }
    case MessageType::PYTHON_RREF_FETCH_CALL: {
      return PythonRRefFetchCall::fromMessage(request);
    }
    case MessageType::RREF_USER_DELETE: {
      return RRefUserDelete::fromMessage(request);
    }
    case MessageType::RREF_CHILD_ACCEPT: {
      return RRefChildAccept::fromMessage(request);
    }
    case MessageType::RREF_FORK_REQUEST: {
      return RRefForkRequest::fromMessage(request);
    }
    case MessageType::FORWARD_AUTOGRAD_REQ: {
      return autograd::RpcWithAutograd::fromMessage(request);
    }
    case MessageType::BACKWARD_AUTOGRAD_REQ: {
      return autograd::PropagateGradientsReq::fromMessage(request);
    }
    case MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ: {
      return autograd::CleanupAutogradContextReq::fromMessage(request);
    }
    case MessageType::RUN_WITH_PROFILING_REQ: {
      return autograd::RpcWithProfilingReq::fromMessage(request);
    }
    case MessageType::RREF_BACKWARD_REQ: {
      return autograd::RRefBackwardReq::fromMessage(request);
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Request type ", request.type(), " not supported.");
    }
  }
}

std::unique_ptr<RpcCommandBase> deserializeResponse(
    const Message& response,
    MessageType& wrappedMsgType) {
  switch (response.type()) {
    case MessageType::SCRIPT_RET: {
      return ScriptResp::fromMessage(response);
    }
    case MessageType::PYTHON_RET: {
      return PythonResp::fromMessage(response);
    }
    case MessageType::REMOTE_RET: {
      return RemoteRet::fromMessage(response);
    }
    case MessageType::SCRIPT_RREF_FETCH_RET: {
      return ScriptRRefFetchRet::fromMessage(response);
    }
    case MessageType::PYTHON_RREF_FETCH_RET: {
      return PythonRRefFetchRet::fromMessage(response);
    }
    case MessageType::RREF_ACK: {
      return RRefAck::fromMessage(response);
    }
    case MessageType::FORWARD_AUTOGRAD_RESP: {
      std::unique_ptr<RpcCommandBase> rpcPtr =
          autograd::RpcWithAutograd::fromMessage(response);
      RpcCommandBase& rpc = *rpcPtr;
      auto& rpcWithAutograd = static_cast<autograd::RpcWithAutograd&>(rpc);

      // Attach 'recv' autograd function.
      addRecvRpcBackward(
          rpcWithAutograd.autogradMetadata(),
          rpcWithAutograd.tensors(),
          rpcWithAutograd.fromWorkerId());

      wrappedMsgType = rpcWithAutograd.wrappedMessageType();

      return std::move(rpcWithAutograd).moveWrappedRpc();
    }
    case MessageType::BACKWARD_AUTOGRAD_RESP: {
      return autograd::PropagateGradientsResp::fromMessage(response);
    }
    case MessageType::CLEANUP_AUTOGRAD_CONTEXT_RESP: {
      return autograd::CleanupAutogradContextResp::fromMessage(response);
    }
    case MessageType::RUN_WITH_PROFILING_RESP: {
      std::unique_ptr<RpcCommandBase> rpcPtr =
          autograd::RpcWithProfilingResp::fromMessage(response);
      RpcCommandBase& rpc = *rpcPtr;
      auto& rpcWithProfilingResp =
          static_cast<autograd::RpcWithProfilingResp&>(rpc);
      // Process remotely profiled events.
      processRemoteProfiledEvents(rpcWithProfilingResp);

      wrappedMsgType = rpcWithProfilingResp.wrappedMessageType();
      auto wrappedRPC = std::move(rpcWithProfilingResp).moveWrappedRpc();
      return wrappedRPC;
    }
    case MessageType::RREF_BACKWARD_RESP: {
      return autograd::RRefBackwardResp::fromMessage(response);
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Response type ", response.type(), " not supported.");
    }
  }
}

IValue deserializeResptoIValueInternal(
    RpcCommandBase& rpc,
    MessageType messageType) {
  switch (messageType) {
    case MessageType::SCRIPT_RET: {
      auto& ret = static_cast<ScriptResp&>(rpc);
      return ret.value();
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false,
          "Response type ",
          messageType,
          " is not supported to be deserialized to IValue.");
    }
  }
}

IValue deserializeRespToIValue(const Message& message) {
  MessageType msgType = message.type();
  auto response = deserializeResponse(message, msgType);
  return deserializeResptoIValueInternal(*response, msgType);
}

namespace {

// Helper for wireDeserialize() below.
//
// The format we use below looks like:
//    section_name_1 size_1\n
//    section_name_2 size_2\n
//    ..
//    \n
//    [sections in order]
//
// Sections themselves include:
//    - "payload" - the payload bits
//    - "meta"    - metadata for the unpickler
//    - "0" ...   - tensor sections for the unpickler
//
// Note that per the header comments, the format is subject to change,
// and is best used for rpcs, rather than persistent disk storage.
std::unordered_map<std::string, std::pair<const char*, size_t>>
parseWireSections(const void* data, size_t data_size) {
  const char* ptr = static_cast<const char*>(data);
  const char* endp = ptr + data_size;

  std::vector<std::pair<std::string, size_t>> headerEnts;
  bool ok = false;
  while (ptr != endp) {
    if (*ptr == '\n') {
      ok = true; // The only "correct" exit point.
      ++ptr;
      break;
    }
    // Parse name
    const char* namePtr = ptr;
    while (ptr != endp && *ptr != ' ') {
      ptr++;
    }
    if (ptr == endp) {
      break;
    }
    std::string name(namePtr, ptr - namePtr);
    if (++ptr == endp) {
      break; // past the ' '
    }
    // Parse size
    const char* sizePtr = ptr;
    while (ptr != endp && *ptr != '\n') {
      ptr++;
    }
    if (ptr == endp) {
      break;
    }
    size_t sz = c10::stoll(std::string(sizePtr, ptr - sizePtr));
    headerEnts.emplace_back(std::make_pair(name, sz));
    ++ptr; // past the '\n'
  }
  if (!ok) {
    throw std::runtime_error("failed parse");
  }

  std::unordered_map<std::string, std::pair<const char*, size_t>> out;
  for (const auto& headerEnt : headerEnts) {
    out[headerEnt.first] = {ptr, headerEnt.second};
    ptr += headerEnt.second;
  }
  if (ptr != endp) {
    throw std::runtime_error("failed bounds");
  }
  return out;
}

static const char* kMeta = "meta";
static const char* kPayload = "payload";
}; // namespace

c10::List<at::Tensor> cloneSparseTensors(
    const std::vector<at::Tensor>& tensors) {
  // Sanity-check: If the majority of bits don't need to go over the wire,
  // force a clone(). Some Tensors are effectively small views, only using
  // ~1% of the underlying Storage.
  auto worthRecopying = [](const at::Tensor& t) -> bool {
    if (!t.has_storage()) {
      return false; // avoid throwing below.
    }
    auto storageSize = t.storage().nbytes();
    auto usefulSize = t.element_size() * t.numel();
    constexpr size_t kMinMultiple = 2;
    constexpr size_t kMinRecopyBytes = 8 * 1024;
    return storageSize >= kMinRecopyBytes &&
        storageSize >= usefulSize * kMinMultiple;
  };
  c10::List<at::Tensor> pTensors;
  pTensors.reserve(tensors.size());
  for (const auto& t : tensors) {
    pTensors.push_back(worthRecopying(t) ? t.clone() : t);
  }
  return pTensors;
}

std::string wireSerialize(
    const std::vector<char>& payload,
    const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    TORCH_CHECK(
        tensor.device().is_cpu(),
        "ProcessGroup RPC backend only supports",
        " CPU tensors, please move your tensors to CPU before sending ",
        "them over RPC. Found tensor on device: ",
        tensor.device());
  }

  struct Ent {
    std::string name;
    const char* data;
    size_t size;
  };
  std::vector<Ent> entries;
  std::string metaEntry;
  std::vector<at::Tensor> tensorData;

  if (!payload.empty()) {
    entries.push_back({kPayload, payload.data(), payload.size()});
  }

  if (!tensors.empty()) {
    torch::jit::Pickler pickler([&](const void* buf, size_t sz) -> size_t {
      metaEntry.append(static_cast<const char*>(buf), sz);
      return sz;
    });
    pickler.protocol();
    pickler.pushIValue(cloneSparseTensors(tensors));
    pickler.stop();
    tensorData = pickler.tensorData();
    entries.push_back({kMeta, metaEntry.data(), metaEntry.size()});
    for (size_t i = 0; i < tensorData.size(); i++) {
      // Construct WritableTensorData for each tensor in the pickler tensorData
      // Since tensorData is in function scope, and getWritableTensorData just
      // record the tensors, the data() pointers stay valid for CPU tensors
      // Note that RPC serde doesn't support CUDA tensors yet, if we should
      // support CUDA tensor, we need to be careful since getWritableTensorData
      // converts CUDA tensor to cpu and data() might get destructed as we go
      // out of scope of this loop.
      auto writeableTensorData = jit::getWriteableTensorData(tensorData[i]);
      entries.push_back({c10::to_string(i),
                         writeableTensorData.data(),
                         writeableTensorData.sizeInBytes()});
    }
  }

  std::string header;
  size_t tot = 0;
  for (const auto& e : entries) {
    tot += e.size;
    header.append(e.name)
        .append(" ")
        .append(c10::to_string(e.size))
        .append("\n");
  }
  header.push_back('\n');

  std::string out;
  out.reserve(header.size() + tot);
  out.append(header);
  for (const auto& e : entries) {
    out.append(e.data, e.size);
  }
  return out;
}

std::pair<std::vector<char>, std::vector<at::Tensor>> wireDeserialize(
    const void* data,
    size_t data_size) {
  auto sections = parseWireSections(data, data_size);

  std::vector<char> payload;
  auto payloadIt = sections.find(kPayload);
  if (payloadIt != sections.end() && payloadIt->second.second != 0) {
    payload.assign(
        payloadIt->second.first,
        payloadIt->second.first + payloadIt->second.second);
  }

  std::vector<at::Tensor> tensors;
  auto metaIt = sections.find(kMeta);
  if (metaIt != sections.end()) {
    const auto& metaData = metaIt->second;
    size_t metaDataPos = 0;
    auto metaDataReadFunc = [&](char* buf, size_t n) -> size_t {
      if (metaDataPos >= metaData.second || n == 0) {
        return 0;
      }
      size_t toCopy = std::min(metaDataPos + n, metaData.second) - metaDataPos;
      memcpy(buf, metaData.first + metaDataPos, toCopy);
      metaDataPos += toCopy;
      return toCopy;
    };
    auto sectionReadFunc = [&](const std::string& ename) -> at::DataPtr {
      auto it = sections.find(ename);
      if (it == sections.end()) {
        throw std::runtime_error("Couldn't find entity " + ename);
      }
      const auto& idat = it->second;
      auto dptr = at::getCPUAllocator()->allocate(idat.second);
      if (idat.second != 0) {
        memcpy(dptr.get(), idat.first, idat.second);
      }
      return dptr;
    };

    // No need to pass typeResolver here, as it always processes string and
    // tensors only
    torch::jit::Unpickler unpickler(
        metaDataReadFunc, nullptr, nullptr, sectionReadFunc, {});
    auto ival = unpickler.parse_ivalue();
    for (auto&& t : ival.toTensorList()) {
      tensors.emplace_back(std::move(t));
    }
  }
  return {std::move(payload), std::move(tensors)};
}

void writeWrappedPayload(
    std::vector<char>& originalPayload,
    std::vector<char>& additionalPayload) {
  originalPayload.insert(
      originalPayload.end(),
      additionalPayload.begin(),
      additionalPayload.end());

  // Add size of the additional payload
  int64_t indexToWrite = originalPayload.size();
  originalPayload.resize(originalPayload.size() + sizeof(int64_t));
  const int64_t additionalPayloadSize = additionalPayload.size();
  torch::utils::THP_encodeInt64Buffer(
      reinterpret_cast<uint8_t*>(originalPayload.data()) + indexToWrite,
      &additionalPayloadSize,
      torch::utils::THPByteOrder::THP_BIG_ENDIAN,
      1);
}

std::vector<at::IValue> readWrappedPayload(
    std::vector<char>& payload,
    const rpc::Message& message) {
  // Read the additional payload remove it from the payload.
  int64_t additionalPayloadSize;
  size_t indexToRead = payload.size() - sizeof(int64_t);
  TORCH_INTERNAL_ASSERT(indexToRead >= 0);
  torch::utils::THP_decodeInt64Buffer(
      &additionalPayloadSize,
      reinterpret_cast<uint8_t*>(payload.data()) + indexToRead,
      torch::utils::THPByteOrder::THP_BIG_ENDIAN,
      1);
  payload.resize(indexToRead);

  TORCH_INTERNAL_ASSERT(
      payload.size() > additionalPayloadSize,
      "Wrong payload sizes: payload.size() is ",
      payload.size(),
      " but additional payload size is ",
      additionalPayloadSize);
  auto wrappedPayloadBegin =
      static_cast<const char*>(message.payload().data()) + payload.size() -
      additionalPayloadSize;
  std::vector<torch::Tensor> tensorTable;
  IValue tuple = jit::unpickle(
      wrappedPayloadBegin,
      additionalPayloadSize,
      *rpc::RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      &tensorTable);
  std::vector<at::IValue> tupleElements = tuple.toTuple()->elements();
  payload.resize(payload.size() - additionalPayloadSize);
  return tupleElements;
}

void populateRemoteProfiledEvents(
    std::vector<torch::autograd::profiler::LegacyEvent>& profiledEvents,
    const torch::autograd::profiler::ProfilerConfig& profilingConfig,
    const std::vector<std::vector<torch::autograd::profiler::LegacyEvent>>&
        eventLists) {
  // Gather all events into a vector
  for (auto& l : eventLists) {
    for (auto& e : l) {
      profiledEvents.push_back(e);
    }
  }
  // find __start_profile event and __cuda_start_event.
  bool cudaProfilingEnabled =
      profilingConfig.state == torch::autograd::profiler::ProfilerState::CUDA;
  bool foundCpuStart = false;
  const torch::autograd::profiler::LegacyEvent* profilerStart = nullptr;
  // Each device has its own cudaProfilerStart, so we must take
  // care to use the correct one depending on the device the
  // operation ran on.
  std::unordered_map<int, const torch::autograd::profiler::LegacyEvent*>
      cudaProfilerStarts;
  for (auto& e : profiledEvents) {
    if (!foundCpuStart && 0 == strcmp(e.name(), "__start_profile")) {
      profilerStart = &e;
      foundCpuStart = true;
    } else if (
        cudaProfilingEnabled && 0 == strcmp(e.name(), "__cuda_start_event")) {
      e.setCudaUs(e.cpuUs());
      auto device = e.device();
      TORCH_CHECK(
          device != -1,
          "CUDA profiling was enabled but could not find CUDA device.");
      TORCH_CHECK(
          cudaProfilerStarts.find(device) == cudaProfilerStarts.end(),
          c10::str("Duplicate __cuda_start_event found for ", device));
      cudaProfilerStarts[device] = &e;
    }

    // TODO: determine no. of CUDA devices and break here if we have
    // a cudaProfilerStart for all of them, in the case of cuda
    // profiling.
    if (foundCpuStart && !cudaProfilingEnabled) {
      break;
    }
  }
  // We should always find __start_profile.
  TORCH_CHECK(
      profilerStart != nullptr, "Expected to find __start_profile event.");
  // Should have >= 1 CUDA start event if cudaProfilingEnabled.
  // TODO: we can enhance this assert by ensuring we have found a
  // start for every available CUDA device.
  TORCH_CHECK(
      !cudaProfilingEnabled || cudaProfilerStarts.size() > 0,
      "Profiler was enabled with CUDA recording, but did not find __cuda_start_event.");

  if (cudaProfilingEnabled) {
    // Compute and set global time for when this CUDA kernel was
    // launched/ended, since deserialized event will not have a
    // corresponding CUDA event.
    for (auto& e : profiledEvents) {
      if (e.hasCuda()) {
        auto cudaDevice = e.device();
        TORCH_CHECK(
            cudaDevice != -1,
            "CUDA profiling was enabled but could not find CUDA device.");
        auto it = cudaProfilerStarts.find(cudaDevice);
        TORCH_CHECK(
            it != cudaProfilerStarts.end(),
            c10::str(
                "Failed to find __cuda_start_event for device ", cudaDevice));
        auto cudaProfilerStartEvent = it->second;
        double cudaElapsedUs = cudaProfilerStartEvent->cudaElapsedUs(e);
        int64_t cudaUs = cudaElapsedUs + cudaProfilerStartEvent->cpuUs();
        e.setCudaUs(cudaUs);
      }
    }
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
