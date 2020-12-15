#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>

// Use of c10::hip namespace here makes hipification easier, because
// I don't have to also fix namespaces.  Sorry!
namespace c10 { namespace hip {

// Takes a valid HIPAllocator (of any sort) and turns it into
// an allocator pretending to be a CUDA allocator.  See
// Note [Masquerading as CUDA]
class HIPAllocatorMasqueradingAsCUDA final : public Allocator {
  Allocator* allocator_;
public:
  explicit HIPAllocatorMasqueradingAsCUDA(Allocator* allocator)
    : allocator_(allocator) {}
  DataPtr allocate(size_t size) const override {
    DataPtr r = allocator_->allocate(size);
    r.unsafe_set_device(Device(DeviceType::CUDA, r.device().index()));
    return r;
  }
  DeleterFnPtr raw_deleter() const override {
    return allocator_->raw_deleter();
  }
};

}} // namespace c10::hip
