#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// This struct defines shader, and shader layout, caches intended to minimize
// redundant object reconstructions at the cost of extra memory consumption.
//
// A shader is a small, usually simple, program that typically runs on a GPU as
// part of the graphics or compute pipelines.  The shader layout defines the
// interface between that program and the outside world, namely what the host
// (i.e. CPU) sees as configurable parameters of the said shader per dispatch.
// If the shader was a regular function, the shader layout would have been its
// function prototype declaring the number and type of its arguments.
//
// Furthermore, shader layouts, or as Vulkan calls them descriptor set layouts,
// define the blueprint out of which descriptor sets are instantiated.  Descriptor
// sets themselves, bundle the input to and output from a shader and contain
// pointers to GPU, and GPU accessible system, memory locations where the actual
// resources reside.  Shader layouts are also used in creation of Vulkan pipeline
// layouts, while multiple shaders are bundled together to form a portion of the
// the monolithic state objects that are Vulkan pipelines.
//
// This struct defines the facilities required to create, compile, reuse,
// and destruct the aforementioned Vulkan objects.
//

struct Shader final {
  //
  // Layout
  //

  struct Layout final {
    /*
      Signature
    */

    typedef c10::SmallVector<VkDescriptorType, 6u> Signature;

    /*
      Descriptor
    */

    struct Descriptor final {
      Signature signature;
    };

    /*
      Factory
    */

    class Factory final {
     public:
      explicit Factory(const GPU& gpu);

      typedef Layout::Descriptor Descriptor;
      typedef VK_DELETER(DescriptorSetLayout) Deleter;
      typedef Handle<VkDescriptorSetLayout, Deleter> Handle;

      struct Hasher {
        size_t operator()(const Descriptor& descriptor) const;
      };

      Handle operator()(const Descriptor& descriptor) const;

     private:
      VkDevice device_;
    };

    struct Object final {
      VkDescriptorSetLayout handle;
      Signature signature;

      operator bool() const;
    };

    /*
      Cache
    */

    class Cache final {
     public:
      explicit Cache(Factory factory);
      Cache(const Cache&) = delete;
      Cache& operator=(const Cache&) = delete;
      Cache(Cache&&) = default;
      Cache& operator=(Cache&&) = default;
      ~Cache() = default;

      Object retrieve(const Descriptor& descriptor);
      void purge();

     private:
      api::Cache<Factory> cache_;
    } cache;

    explicit Layout(const GPU& gpu)
      : cache(Factory(gpu)) {
    }
  } layout;

  //
  // Work Group
  //

  typedef utils::uvec3 WorkGroup;

  /*
    Descriptor
  */

  struct Descriptor final {
    enum class Type {
      Source,
      Binary,
    } type;

    union {
      struct {
        const char* glsl; // Null-terminated
        uint32_t unused;  // Padding
      } source;

      struct {
        const uint32_t* spirv;
        uint32_t size;    // Bytes
      } binary;
    } shader;

    Descriptor(const char* glsl);
    Descriptor(const uint32_t* spirv, uint32_t bytes);
  };

  /*
    Factory
  */

  class Factory final {
   public:
    explicit Factory(const GPU& gpu);
    Factory(const Factory&) = delete;
    Factory& operator=(const Factory&) = delete;
    Factory(Factory&&);
    Factory& operator=(Factory&&);
    ~Factory();

    typedef Shader::Descriptor Descriptor;
    typedef VK_DELETER(ShaderModule) Deleter;
    typedef Handle<VkShaderModule, Deleter> Handle;

    struct Hasher {
      size_t operator()(const Descriptor& descriptor) const;
    };

    Handle operator()(const Descriptor& descriptor) const;

   private:
    VkDevice device_;
    struct Compiler;
    std::unique_ptr<Compiler> compiler_;
  };

  /*
    Cache
  */

  typedef api::Cache<Factory> Cache;
  Cache cache;

  explicit Shader(const GPU& gpu)
    : layout(gpu),
      cache(Factory(gpu)) {
  }
};

//
// Impl
//

inline bool operator==(
    const Shader::Layout::Descriptor& _1,
    const Shader::Layout::Descriptor& _2) {
  return _1.signature == _2.signature;
}

inline size_t Shader::Layout::Factory::Hasher::operator()(
    const Descriptor& descriptor) const {
  size_t hash = 0u;

  for (const VkDescriptorType type : descriptor.signature) {
    hash = c10::hash_combine(
        hash,
        c10::get_hash(type));
  }

  return hash;
}

inline Shader::Layout::Object::operator bool() const {
  return VK_NULL_HANDLE != handle;
}

inline Shader::Layout::Object Shader::Layout::Cache::retrieve(
    const Descriptor& descriptor) {
  return {
    cache_.retrieve(descriptor),
    descriptor.signature,
  };
}

inline void Shader::Layout::Cache::purge() {
  cache_.purge();
}

inline bool operator==(
    const Shader::WorkGroup& _1,
    const Shader::WorkGroup& _2) {
  return (_1.data[0u] == _2.data[0u]) &&
         (_1.data[1u] == _2.data[1u]) &&
         (_1.data[2u] == _2.data[2u]);
}

inline Shader::Descriptor::Descriptor(const char* const glsl)
 : type(Type::Source),
   shader{
    .source = {
      glsl,
      0u,
    },
   } {
  TORCH_CHECK(glsl, "Invalid shader source code!");
}

inline Shader::Descriptor::Descriptor(
    const uint32_t* const code,
    const uint32_t size)
 : type(Type::Binary),
   shader{
    .binary = {
      code,
      size,
    },
   } {
  TORCH_CHECK(code && (0u != size), "Invalid shader binary!");
}

inline bool operator==(
    const Shader::Descriptor& _1,
    const Shader::Descriptor& _2) {
  static_assert(
      sizeof(Shader::Descriptor::shader.source) == sizeof(Shader::Descriptor::shader.binary),
      "This implementation requires sizeof(Source) to be equal to sizeof(Binary).");

  return (_1.type == _2.type) &&
         (_1.shader.binary.spirv == _2.shader.binary.spirv) &&
         (_1.shader.binary.size == _2.shader.binary.size);
}

inline size_t Shader::Factory::Hasher::operator()(
    const Descriptor& descriptor) const {
  static_assert(
      sizeof(Descriptor::shader.source) == sizeof(Descriptor::shader.binary),
      "This implementation requires sizeof(Source) to be equal to sizeof(Binary).");

  return c10::get_hash(
      descriptor.type,
      descriptor.shader.binary.spirv,
      descriptor.shader.binary.size);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

inline bool operator==(
    const VkDescriptorSetLayoutBinding& _1,
    const VkDescriptorSetLayoutBinding& _2) {
  return (_1.binding == _2.binding) &&
         (_1.descriptorType == _2.descriptorType) &&
         (_1.descriptorCount == _2.descriptorCount) &&
         (_1.stageFlags == _2.stageFlags) &&
         (_1.pImmutableSamplers == _2.pImmutableSamplers);
}

#endif /* USE_VULKAN_API */
