#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <exception>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>
#include <regex>

// Check if compiler has working std::regex implementation
//
// Test below is adapted from https://stackoverflow.com/a/41186162
#if defined(_MSVC_LANG) && _MSVC_LANG >= 201103L
  // Compiler has working regex. MSVC has erroneous __cplusplus.
#elif __cplusplus >= 201103L &&                           \
    (!defined(__GLIBCXX__) || (__cplusplus >= 201402L) || \
        (defined(_GLIBCXX_REGEX_DFS_QUANTIFIERS_LIMIT) || \
         defined(_GLIBCXX_REGEX_STATE_LIMIT)           || \
             (defined(_GLIBCXX_RELEASE)                && \
             _GLIBCXX_RELEASE > 4)))
  // Compiler has working regex.
#else
  static_assert(false, "Compiler does not have proper regex support.");
#endif

namespace c10 {
namespace {
DeviceType parse_type(const std::string& device_string) {
  static const std::array<std::pair<std::string, DeviceType>, 11> types = {{
      {"cpu", DeviceType::CPU},
      {"cuda", DeviceType::CUDA},
      {"mkldnn", DeviceType::MKLDNN},
      {"opengl", DeviceType::OPENGL},
      {"opencl", DeviceType::OPENCL},
      {"ideep", DeviceType::IDEEP},
      {"hip", DeviceType::HIP},
      {"fpga", DeviceType::FPGA},
      {"msnpu", DeviceType::MSNPU},
      {"xla", DeviceType::XLA},
      {"vulkan", DeviceType::Vulkan},
  }};
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [device_string](const std::pair<std::string, DeviceType>& p) {
        return p.first == device_string;
      });
  if (device != types.end()) {
    return device->second;
  }
  AT_ERROR(
      "Expected one of cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu, xla, vulkan device type at start of device string: ", device_string);
}
} // namespace

Device::Device(const std::string& device_string) : Device(Type::CPU) {
  TORCH_CHECK(!device_string.empty(), "Device string must not be empty");

  // We assume gcc 5+, so we can use proper regex.
  static const std::regex regex("([a-zA-Z_]+)(?::([1-9]\\d*|0))?");
  std::smatch match;
  TORCH_CHECK(
    std::regex_match(device_string, match, regex),
    "Invalid device string: '", device_string, "'");
  type_ = parse_type(match[1].str());
  if (match[2].matched) {
    try {
      index_ = c10::stoi(match[2].str());
    } catch (const std::exception &) {
      AT_ERROR(
        "Could not parse device index '", match[2].str(),
        "' in device string '", device_string, "'");
    }
  }
  validate();
}

std::string Device::str() const {
  std::string str = DeviceTypeName(type(), /* lower case */ true);
  if (has_index()) {
    str.push_back(':');
    str.append(to_string(index()));
  }
  return str;
}

std::ostream& operator<<(std::ostream& stream, const Device& device) {
  stream << device.str();
  return stream;
}

} // namespace c10
