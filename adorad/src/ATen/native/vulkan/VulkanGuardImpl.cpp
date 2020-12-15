#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

namespace at {
namespace detail {

struct VulkanGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  VulkanGuardImpl() {}

  explicit VulkanGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::Vulkan);
  }

  DeviceType type() const override {
    return DeviceType::Vulkan;
  }
  Device exchangeDevice(Device) const override {
    // no-op
    return Device(DeviceType::Vulkan, -1);
  }
  Device getDevice() const override {
    return Device(DeviceType::Vulkan, -1);
  }
  void setDevice(Device) const override {
    // no-op
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    // no-op
  }
  Stream getStream(Device d) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(DeviceType::Vulkan, -1));
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(DeviceType::Vulkan, -1));
  }
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }

  // Event-related functions
  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    TORCH_CHECK(false, "VULKAN backend doesn't support events.");
  }
  void block(void* event, const Stream& stream) const override {
    TORCH_CHECK(false, "VULKAN backend doesn't support events.")
  }
  bool queryEvent(void* event) const override {
    TORCH_CHECK(false, "VULKAN backend doesn't support events.")
  }
  void destroyEvent(void* event, const DeviceIndex device_index) const
      noexcept override {}
};

C10_REGISTER_GUARD_IMPL(Vulkan, VulkanGuardImpl);

} // namespace detail
} // namespace at
