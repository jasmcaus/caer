#pragma once

#include <ATen/core/TensorBody.h>
#include <ATen/core/blob.h>
#include <c10/util/C++17.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <typeindex>

namespace torch {
class TORCH_API CustomClassHolder : public c10::intrusive_ptr_target {};
namespace jit {
using ::torch::CustomClassHolder;
struct Function;
struct CompilationUnit;
struct Module;
} // namespace jit
} // namespace torch
namespace c10 {
template <class Key, class Value>
class Dict;
template <class T>
class List;
struct IValue;
struct ClassType;
struct Type;
class RRefInterface;
using TypePtr = std::shared_ptr<Type>;

struct ClassType;
using ClassTypePtr = std::shared_ptr<ClassType>;

TORCH_API bool _fastEqualsForContainer(const IValue& lhs, const IValue& rhs);

TORCH_API torch::jit::Function* checkObjectSortSchema(
    const c10::ClassTypePtr& t,
    std::stringstream& why_not);

// A comparator that checks ordering of two IValues of same type.
typedef std::function<bool(const IValue& a, const IValue& b)> IValueComparator;

TORCH_API IValueComparator getLessThanComparator(const IValue& v);
TORCH_API IValueComparator getGreaterThanComparator(const IValue& v);

namespace ivalue {
struct Tuple;
struct Future;
struct ConstantString;
struct GenericDict;
struct Object;
struct PyObjectHolder;
struct EnumHolder;
} // namespace ivalue

// This is an owning wrapper for a c10::optional<std::vector<T>>
// that can be implicitly converted to a (non-owning) optional<ArrayRef<T>>.
// Its purpose is to be used in generated code to keep the vector alive
// either until the end of a statement (as a temporary), or as a saved arg
// in autograd.
template <typename T>
struct OptionalArray {
  c10::optional<std::vector<T>> list;

  OptionalArray(){}
  OptionalArray(std::vector<T> val) : list(std::move(val)) {}

  // Used when saving an argument for the backwards pass.
  OptionalArray& operator=(c10::optional<ArrayRef<T>> ref) {
    if (ref) {
      list = std::vector<T>(ref->begin(), ref->end());
    } else {
      list = nullopt;
    }
    return *this;
  }

  operator c10::optional<c10::ArrayRef<T>>() {
    if (!list) {
      return nullopt;
    }
    return *list;
  }
};

// Capsule is an internal implementation detail of custom C++ classes. We
// define it as an owning wrapper for
// c10::intrusive_ptr<torch::CustomClassHolder> This wrapper is here to serve as
// an abstraction of the type erased custom class object pointer. It also allow
// pybind11 to treat this as a standalone class to register as a separate type
// caster, instead of a custom pointer holder which the pointer holder type
// caster try to "unwrap" it automatically.
struct Capsule {
  c10::intrusive_ptr<torch::CustomClassHolder> obj_ptr;
  explicit Capsule(c10::intrusive_ptr<torch::CustomClassHolder> ptr)
      : obj_ptr(std::move(ptr)) {}
};

// IValue is the generic tagged union used by the interpreter to hold
// all value types.
// It is a 16-byte object with an 8-byte payload and an 8-byte tag.
// The tag is currently 4 bytes to determine the type, and 1 byte
// to mark whether that type is a subtype of c10::intrusive_ptr_target and needs
// retain/release calls.

#define TORCH_FORALL_TAGS(_) \
  _(None)                    \
  _(Tensor)                  \
  _(Double)                  \
  _(Int)                     \
  _(Bool)                    \
  _(Tuple)                   \
  _(String)                  \
  _(Blob)                    \
  _(GenericList)             \
  _(GenericDict)             \
  _(Future)                  \
  _(Device)                  \
  _(Stream)                  \
  _(Object)                  \
  _(PyObject)                \
  _(Uninitialized)           \
  _(Capsule)                 \
  _(RRef)                    \
  _(Quantizer)               \
  _(Generator)               \
  _(Enum)

// [doxygen private]
// These methods are not actually private but we don't want to document them, so
// they are marked `@private`, which hides them on the doxygen documentation for
// this page.

/// IValue (Interpreter Value) is a tagged union over the types supported by the
/// TorchScript interpreter. IValues contain their values as an
/// `IValue::Payload`, which holds primitive types (`int64_t`, `bool`, `double`,
/// `Device`), as values and all other types as a `c10::intrusive_ptr`.
///
/// IValues are used as inputs to and outputs from the TorchScript interpreter.
/// To retrieve the value contained within an IValue, use the `.toX()` methods,
/// where `X` is the type you are trying to get. Note that neither the `.toX()`
/// methods nor the templated `.to<T>` functions do any kind of casting, they
/// only unwrap the contained value. For example:
///
/// \rst
/// .. code-block:: cpp
///
///   // Make the IValue
///   torch::IValue my_ivalue(26);
///   std::cout << my_ivalue << "\n";
///
///   // Unwrap the IValue
///   int64_t my_int = my_ivalue.toInt()
///   std::cout << my_int << "\n";
///
///   // This will throw an error!
///   // `my_ivalue` is tagged as an int and cannot be used as another type
///   torch::Tensor my_tensor = my_ivalue.toTensor()
/// \endrst
struct CAFFE2_API IValue final {
  IValue(const IValue& rhs)
      : IValue(rhs.payload, rhs.tag, rhs.is_intrusive_ptr) {
    if (is_intrusive_ptr) {
      c10::raw::intrusive_ptr::incref(payload.as_intrusive_ptr);
    }
  }
  IValue(IValue&& rhs) noexcept : IValue() {
    swap(rhs);
  }
  /// @private [doxygen private]
  ~IValue() {
    if (is_intrusive_ptr) {
      c10::raw::intrusive_ptr::decref(payload.as_intrusive_ptr);
    }
  }
  IValue& operator=(IValue&& rhs) & noexcept {
    IValue(std::move(rhs)).swap(*this); // this also sets rhs to None
    return *this;
  }
  IValue& operator=(IValue const& rhs) & {
    IValue(rhs).swap(*this);
    return *this;
  }
  void dump() const;

  /**
   * Equality comparison. The semantics are the same as Python's `==`:
   * 1. Numerical types are compared by value.
   * 2. Tensors compute element-wise equality, returning a BoolTensor (see:
   * `torch.eq()`)
   * 3. Strings are compared by value.
   * 4. Sequence types (list, tuple) are compared lexicographically by
   *    comparing their elements. Different sequence types never compare equal.
   * 5. Mappings (dict) must have equal (key, value) pairs.
   * 6. If not listed above, the default behavior for is to test identity
   * equality (e.g. pointer equality).
   *
   * Why does this return an IValue instead of a bool? Because in PyTorch,
   * `tensor1 == tensor2` returns a `BoolTensor`, not a bool.
   *
   * NOTE: we (like Python) assume that identity equality implies value equality
   * for efficiency.
   * TODO: need to support customizing equality
   */
  IValue equals(const IValue& rhs) const;
  /**
   * This implements the same semantics as `bool(lhs == rhs)` in Python. which
   * is the same as `equals()` except for Tensor types.
   */
  TORCH_API friend bool operator==(const IValue& lhs, const IValue& rhs);
  TORCH_API friend bool operator!=(const IValue& lhs, const IValue& rhs);

  /**
   * Identity comparison. Checks if `this` is the same object as `rhs`. The
   * semantics are the same as Python's `is` operator.
   *
   * NOTE: Like in Python, this operation is poorly defined for primitive types
   * like numbers and strings. Prefer to use `==` unless you really want to
   * check identity equality.
   */
  bool is(const IValue& rhs) const;

   /**
   * Hashing for IValues. Returns an IValue-boxed int.
   *
   * Some notes:
   * - Like eager, Tensors are hashed by looking at the pointer. This is not
   *   strictly correct because two value-equal tensors with different tensor
   *   pointers will hash differently, but we choose to reproduce the eager
   *   semantics.
   * - Hashing is not defined on all built-in IValue types (e.g. list and
   *   dict), following Python. Calling `hash()` on these types will throw.
   */
  IValue hash() const {
    return (int64_t)IValue::hash(*this);
  }
  // This is defined because `c10::hash` dispatches to a function of this
  // signature. See the member function `hash()`.
  static size_t hash(const IValue& iv);

  /**
   * @private [doxygen private]
   * [container equality]
   * This is an equality implementation that assumes objects with the same
   * identity equal themselves, for efficiency reasons. We primarily have this
   * for consistency, because Python does the same thing. This actually
   * provokes user-visible changes in behavior due to quirks in torch:
   *      [tensor1] == [tensor1] -> True (because container equality will first
   * compare identity) [tensor1] == [tensor1_copy] -> RuntimeError: bool value
   * of Tensor is ambiguous
   */
  TORCH_API friend bool _fastEqualsForContainer(
      const IValue& lhs,
      const IValue& rhs);

  /// @private [doxygen private]
  bool isAliasOf(const IValue& rhs) const {
    if (this->tag != rhs.tag) {
      // Trivially don't alias if the type is different
      return false;
    }

    if (!this->is_intrusive_ptr) {
      // Primitive types don't alias anything
      return false;
    }

    AT_ASSERT(rhs.is_intrusive_ptr);

    // Tensors should be compared based on internal storage
    if (this->isTensor()) {
      const auto thisTensor = this->toTensor();
      const auto rhsTensor = rhs.toTensor();
      return thisTensor.is_alias_of(rhsTensor);
    }

    // Other types can be compared by their ptr value
    return this->payload.as_intrusive_ptr == rhs.payload.as_intrusive_ptr;
  }

  /// @private [doxygen private]
  size_t use_count() const noexcept {
    if (!is_intrusive_ptr) {
      return 1;
    }

    return c10::raw::intrusive_ptr::use_count(payload.as_intrusive_ptr);
  }

  /// @private [doxygen private]
  void swap(IValue& rhs) noexcept {
    std::swap(payload, rhs.payload);
    std::swap(is_intrusive_ptr, rhs.is_intrusive_ptr);
    std::swap(tag, rhs.tag);
  }

  // Accessors for subtypes are arranged together below
  // While some of these accessors could be generated through templates,
  // we prefer to write them manually for clarity

  IValue(at::Tensor t) : tag(Tag::Tensor), is_intrusive_ptr(t.defined()) {
    // Note: the undefined tensor is not refcounted, so while it
    // is tagged as a tensor, is_intrusive_ptr is set to false.
    // This is not an optional optimization: our incref call
    // *will not* do the right thing when called on an
    // undefined tensor.
    payload.as_intrusive_ptr = t.unsafeReleaseTensorImpl();
  }
  bool isTensor() const {
    return Tag::Tensor == tag;
  }
  at::Tensor toTensor() &&;
  at::Tensor toTensor() const&;
  at::TensorImpl* unsafeToTensorImpl() const {
    return static_cast<at::TensorImpl*>(payload.as_intrusive_ptr);
  }

  const IValue& toIValue() const {
    return *this;
  }
  IValue& toIValue() {
    return *this;
  }

  /// @private [doxygen private]
  IValue(intrusive_ptr<caffe2::Blob> blob)
      : tag(Tag::Blob), is_intrusive_ptr(true) {
    // TODO (after Tensor merge) If we pass in a Blob holding a Tensor, extract
    // and store it as a Tensor instead.
    payload.as_intrusive_ptr = blob.release();
  }

  /// @private [doxygen private]
  bool isBlob() const {
    return Tag::Blob == tag;
  }

  /// @private [doxygen private]
  c10::intrusive_ptr<caffe2::Blob> toBlob() &&;

  /// @private [doxygen private]
  c10::intrusive_ptr<caffe2::Blob> toBlob() const&;

  // Capsule. No new callsites of these APIs should
  // be introduced.
  static inline IValue make_capsule(
      intrusive_ptr<torch::CustomClassHolder> blob);
  bool isCapsule() const {
    return Tag::Capsule == tag;
  }
  c10::intrusive_ptr<torch::CustomClassHolder> toCapsule() &&;
  c10::intrusive_ptr<torch::CustomClassHolder> toCapsule() const&;

  // Custom C++ classes
  template <
      typename T,
      std::enable_if_t<
          std::is_base_of<torch::CustomClassHolder, T>::value,
          int> = 0>
  IValue(intrusive_ptr<T> custom_class);
  bool isCustomClass() const;
  template <typename T>
  c10::intrusive_ptr<T> toCustomClass() &&;
  template <typename T>
  c10::intrusive_ptr<T> toCustomClass() const&;

  // Tuple
  IValue(c10::intrusive_ptr<ivalue::Tuple> v);

  template <
      typename... Args,
      std::enable_if_t<
          !guts::disjunction<
              std::is_lvalue_reference<Args>...,
              guts::negation<std::is_constructible<IValue, Args>>...>::value,
          std::nullptr_t> = nullptr>
  IValue(const std::tuple<Args...>& t);
  bool isTuple() const {
    return Tag::Tuple == tag;
  }
  c10::intrusive_ptr<ivalue::Tuple> toTuple() &&;
  c10::intrusive_ptr<ivalue::Tuple> toTuple() const&;

  // Double
  IValue(double d) : tag(Tag::Double), is_intrusive_ptr(false) {
    payload.as_double = d;
  }
  bool isDouble() const {
    return Tag::Double == tag;
  }
  double toDouble() const {
    AT_ASSERT(isDouble());
    return payload.as_double;
  }

  // Future
  IValue(c10::intrusive_ptr<ivalue::Future> v);
  bool isFuture() const {
    return Tag::Future == tag;
  }
  c10::intrusive_ptr<ivalue::Future> toFuture() &&;
  c10::intrusive_ptr<ivalue::Future> toFuture() const&;

  // RRef
  IValue(c10::intrusive_ptr<c10::RRefInterface> v);
  bool isRRef() const {
    return Tag::RRef == tag;
  }
  c10::intrusive_ptr<c10::RRefInterface> toRRef() &&;
  c10::intrusive_ptr<c10::RRefInterface> toRRef() const&;

  // Quantizer
  IValue(c10::intrusive_ptr<at::Quantizer> v);
  bool isQuantizer() const {
    return Tag::Quantizer == tag;
  }
  c10::intrusive_ptr<at::Quantizer> toQuantizer() &&;
  c10::intrusive_ptr<at::Quantizer> toQuantizer() const&;

  // Int
  IValue(int64_t i) : tag(Tag::Int), is_intrusive_ptr(false) {
    payload.as_int = i;
  }

  // allow you to pass literals (3, 4) without ambiguity
  IValue(int32_t i) : IValue(static_cast<int64_t>(i)) {}

  bool isInt() const {
    return Tag::Int == tag;
  }

  int64_t toInt() const {
    AT_ASSERT(isInt());
    return payload.as_int;
  }

  // Bool
  IValue(bool b) : tag(Tag::Bool), is_intrusive_ptr(false) {
#if defined(__clang__) && defined(__x86_64__)
    // Initializing entire payload stops valgrind's from reporting
    // "jump or move depends on uninitialised value" in IValue copy constructor
    // See https://github.com/pytorch/pytorch/issues/37117
    payload.as_int = b;
#else
    payload.as_bool = b;
#endif
  }
  bool isBool() const {
    return Tag::Bool == tag;
  }
  bool toBool() const {
    AT_ASSERT(isBool());
    return payload.as_bool;
  }

  // IntList
  bool isIntList() const;
  c10::List<int64_t> toIntList() &&;
  c10::List<int64_t> toIntList() const&;
  std::vector<int64_t> toIntVector() const;

  // ConstantString
  IValue(c10::intrusive_ptr<ivalue::ConstantString> v);
  IValue(std::string v);
  IValue(const char* v) : IValue(std::string(v)) {}
  bool isString() const {
    return Tag::String == tag;
  }
  c10::intrusive_ptr<ivalue::ConstantString> toString() &&;
  c10::intrusive_ptr<ivalue::ConstantString> toString() const&;
  const std::string& toStringRef() const;
  c10::optional<std::reference_wrapper<const std::string>> toOptionalStringRef()
      const;

  // DoubleList
  bool isDoubleList() const;
  c10::List<double> toDoubleList() &&;
  c10::List<double> toDoubleList() const&;
  std::vector<double> toDoubleVector() const;

  // BoolList
  bool isBoolList() const;
  c10::List<bool> toBoolList() &&;
  c10::List<bool> toBoolList() const&;

  // TensorList
  bool isTensorList() const;
  c10::List<at::Tensor> toTensorList() &&;
  c10::List<at::Tensor> toTensorList() const&;
  std::vector<at::Tensor> toTensorVector() const;

  // GenericList
  IValue(c10::List<IValue> v);
  bool isList() const {
    return Tag::GenericList == tag;
  }
  c10::List<IValue> toList() &&;
  c10::List<IValue> toList() const&;
  c10::ArrayRef<IValue> toListRef() const;

  // Some template constructors of IValue calls another constructor recursively.
  // This SNIFAEs the called constructor exists.
  template <class T>
  using enable_if_ivalue_constructible =
      std::enable_if_t<std::is_constructible<IValue, T>::value, std::nullptr_t>;

  template <class T, enable_if_ivalue_constructible<T> = nullptr>
  IValue(c10::List<T> v);
  template <class T, enable_if_ivalue_constructible<T> = nullptr>
  IValue(at::ArrayRef<T> v);
  template <class T, enable_if_ivalue_constructible<T> = nullptr>
  IValue(const std::vector<T>& v);
  template <class T, size_t N>
  IValue(std::array<T, N> v);

  // GenericDict
  IValue(c10::Dict<IValue, IValue> v);
  bool isGenericDict() const {
    return Tag::GenericDict == tag;
  }
  c10::Dict<IValue, IValue> toGenericDict() &&;
  c10::Dict<IValue, IValue> toGenericDict() const&;

  template <class Key, class Value>
  IValue(c10::Dict<Key, Value> v);

  template <class Key, class Value>
  /// \cond
  /// DOXYGEN_CANNOT_HANDLE_CONSTRUCTORS_WITH_MACROS_SO_EXCLUDE_THIS_LINE_FROM_DOXYGEN
  C10_DEPRECATED_MESSAGE(
      "IValues based on std::unordered_map<K, V> are slow and deprecated. Please use c10::Dict<K, V> instead.")
      /// \endcond
      IValue(std::unordered_map<Key, Value> v);

  template <class T, enable_if_ivalue_constructible<T> = nullptr>
  IValue(c10::optional<T> v);
  IValue(c10::nullopt_t);

  // ClassType
  IValue(c10::intrusive_ptr<ivalue::Object> v);
  bool isObject() const {
    return tag == Tag::Object;
  }
  c10::intrusive_ptr<ivalue::Object> toObject() &&;
  c10::intrusive_ptr<ivalue::Object> toObject() const&;
  const ivalue::Object& toObjectRef() const;

  torch::jit::Module toModule() const;
  bool isModule() const;

  // PyObject
  IValue(c10::intrusive_ptr<ivalue::PyObjectHolder> v);
  bool isPyObject() const {
    return tag == Tag::PyObject;
  }
  c10::intrusive_ptr<ivalue::PyObjectHolder> toPyObjectHolder() &&;
  c10::intrusive_ptr<ivalue::PyObjectHolder> toPyObjectHolder() const&;
  PyObject* toPyObject() const;

  // Enum
  explicit IValue(c10::intrusive_ptr<ivalue::EnumHolder> v);
  bool isEnum() const {
    return tag == Tag::Enum;
  }
  c10::intrusive_ptr<ivalue::EnumHolder> toEnumHolder() &&;
  c10::intrusive_ptr<ivalue::EnumHolder> toEnumHolder() const&;

  // None
  IValue() : payload{0}, tag(Tag::None), is_intrusive_ptr(false) {}
  bool isNone() const {
    return Tag::None == tag;
  }
  std::string toNone() const {
    AT_ASSERT(isNone());
    return "None";
  }

  static IValue uninitialized() {
    auto i = IValue();
    i.tag = Tag::Uninitialized;
    return i;
  }

  // Scalar, which gets encoded as either an Int or a Double
  IValue(at::Scalar s) : IValue() {
    if (s.isFloatingPoint()) {
      *this = s.toDouble();
    } else {
      *this = s.toLong();
    }
  }
  bool isScalar() const {
    return isDouble() || isInt();
  }
  at::Scalar toScalar() const {
    if (isDouble())
      return toDouble();
    else if (isInt())
      return toInt();
    throw std::runtime_error("IValue is not a Scalar");
  }

  // Device
  IValue(c10::Device d) : tag(Tag::Device), is_intrusive_ptr(false) {
    payload.as_device.type = d.type();
    payload.as_device.index = d.index();
  }
  bool isDevice() const {
    return Tag::Device == tag;
  }
  c10::Device toDevice() const {
    AT_ASSERT(isDevice());
    return c10::Device(payload.as_device.type, payload.as_device.index);
  }

  //Stream
  IValue(c10::Stream stream)
    : tag(Tag::Stream), is_intrusive_ptr(false) {
    payload.as_int = stream.pack();
  }
  c10::Stream toStream() &&;
  c10::Stream toStream() const &;
  bool isStream() const { return Tag::Stream == tag; }

  // ScalarType
  IValue(ScalarType t)
      : IValue(static_cast<std::underlying_type<ScalarType>::type>(t)) {}
  at::ScalarType toScalarType() const {
    return static_cast<at::ScalarType>(toInt());
  }

  // Layout
  IValue(Layout l)
      : IValue(static_cast<std::underlying_type<Layout>::type>(l)) {}
  at::Layout toLayout() const {
    return static_cast<at::Layout>(toInt());
  }

  // MemoryFormat
  IValue(MemoryFormat m)
      : IValue(static_cast<std::underlying_type<MemoryFormat>::type>(m)) {}
  at::MemoryFormat toMemoryFormat() const {
    return static_cast<at::MemoryFormat>(toInt());
  }

  // QScheme
  IValue(at::QScheme qscheme) : tag(Tag::Int), is_intrusive_ptr(false) {
    payload.as_int = static_cast<int64_t>(qscheme);
  }

  at::QScheme toQScheme() const {
    return static_cast<at::QScheme>(toInt());
  }

  // Dimname
  IValue(at::Dimname dimname) : IValue(dimname.symbol().toQualString()) {}

  at::Dimname toDimname() const {
    return at::Dimname::fromSymbol(Symbol::fromQualString(toStringRef()));
  }

  // Generator
  IValue(at::Generator g) : tag(Tag::Generator), is_intrusive_ptr(g.defined()) {
    // Note: the undefined generator is not refcounted, so while it
    // is tagged as a generator, is_intrusive_ptr is set to false.
    // This is not an optional optimization: our incref call
    // *will not* do the right thing when called on an
    // undefined generator.
    payload.as_intrusive_ptr = g.unsafeReleaseGeneratorImpl();
  }
  bool isGenerator() const {
    return Tag::Generator == tag;
  }
  at::Generator toGenerator() &&;
  at::Generator toGenerator() const&;

  // for debugging
  std::string tagKind() const {
    switch (tag) {
#define DEFINE_CASE(x) \
  case Tag::x:         \
    return #x;
      TORCH_FORALL_TAGS(DEFINE_CASE)
#undef DEFINE_CASE
    }
    return "InvalidTag(" + c10::guts::to_string(static_cast<int>(tag)) + ")";
  }

  // generic v.to<at::Tensor>() implementations
  // that can be used in special functions like pop/push
  // that use template meta-programming.
  // prefer the directly named methods when you can,
  // since they are simpler to understand

  // Note: if you get linker errors saying one of these is missing,
  // change it to ... && = delete; and you will see better error messages for
  // why However, we cannot commit this because some compiler versions barf on
  // it.
  template <typename T>
  T to() &&;
  template <typename T>
  T to() const&;

  // ToOptional: convert a IValue to the Optional obj that accepts both T and
  // None
  template <typename T>
  optional<T> toOptional();

  /// @private [doxygen private]
  /// Only for use in generated code.
  OptionalArray<int64_t> toOptionalIntArray();

  /// @private [doxygen private]
  /// Only for use in generated code.
  OptionalArray<double> toOptionalDoubleArray();

  /// @private [doxygen private]
  /// this is a shallow comparison of two IValues to test the object identity
  bool isSameIdentity(const IValue& rhs) const;

  // Computes the "official" string representation of an IValue. This produces a
  // TorchScript expression that can be used to recreate an IValue with the same
  // value (e.g. when we are printing constants in the serializer).
  //
  // Callers can use `customFormatter` to override how `repr()` prints out an
  // IValue. This is useful if you have some other environment where you can
  // look up values, and you want to print a reference to that environment (like
  // the serializer's constant table).
  //
  // repr() is not necessarily defined on all objects!
  std::ostream& repr(
      std::ostream& stream,
      std::function<bool(std::ostream&, const IValue& v)> customFormatter)
      const;

  // Computes an "informal" string representation of an IValue. This should be
  // used for debugging, or servicing `print()`-like functions.
  // This is different from `repr()` in that there is no expectation that we can
  // exactly reconstruct an IValue from the output; feel free to use a
  // concise/pretty form
  CAFFE2_API friend std::ostream& operator<<(
      std::ostream& out,
      const IValue& v);

  bool isPtrType() const {
    return is_intrusive_ptr;
  }

  /// @private [doxygen private]
  const void* internalToPointer() const {
    TORCH_INTERNAL_ASSERT(
        isPtrType(), "Can only call internalToPointer() for pointer types");
    return payload.as_intrusive_ptr;
  }

  TypePtr type() const;

  // Detect aliased tensors.
  struct HashAliasedIValue {
    size_t operator()(const IValue& val) const {
      if (val.isTensor()) {
        return reinterpret_cast<size_t>(
            val.toTensor().storage().unsafeGetStorageImpl());
      }
      // If it is not a Tensor, then two mutable IValues alias each other only
      // if they are the same pointer.
      return val.payload.as_int;
    }
  };

  struct CompAliasedIValues {
    bool operator()(const IValue& lhs, const IValue& rhs) const {
      return lhs.isAliasOf(rhs);
    }
  };

  using HashAliasedIValues =
      std::unordered_set<IValue, HashAliasedIValue, CompAliasedIValues>;
  using HashAliasedIValueMap =
      std::unordered_map<IValue, IValue, HashAliasedIValue, CompAliasedIValues>;

  // Chechs if this and rhs has a subvalues in common.
  // [t1,t2] and [t2, t3] returns true.
  bool overlaps(const IValue& rhs) const;

  // Inserts all subvalues of this in subValues.
  void getSubValues(HashAliasedIValues& subValues) const;

  // Apply visitor to every subvalue.
  // TODO: There are several places that recurse over IValue. This is fragile.
  // This visitor should be used to recurse over ivalues.
  void visit(const std::function<bool(const IValue&)>& visitor) const;
  IValue deepcopy() const;
  IValue deepcopy(HashAliasedIValueMap& memo) const;

 private:
  static bool ptrEqual(const IValue& lhs, const IValue& rhs);
  // NOTE: IValue tags are intentionally private. In the future we may encode
  // this value different (e.g. using NaN boxing), and this would make it more
  // costly to determine the tag for all types vs just determining if something
  // is a particular type. Instead we want clients to use the `isX` methods when
  // possible. If for perf. reasons you really, absolutely, must have a jump
  // table, then we can revisit this.
  enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
    TORCH_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
  };

  template <
      class T,
      class NullType = c10::detail::intrusive_target_default_null_type<T>>
  c10::intrusive_ptr<T, NullType> moveToIntrusivePtr();
  template <
      typename T,
      class NullType = c10::detail::intrusive_target_default_null_type<T>>
  c10::intrusive_ptr<T, NullType> toIntrusivePtr() const;

  void clearToNone() {
    payload.as_int = 0;
    tag = Tag::None;
    is_intrusive_ptr = false;
  }

  union Payload {
    int64_t as_int;
    double as_double;
    bool as_bool;
    c10::intrusive_ptr_target* as_intrusive_ptr;
    struct {
      DeviceType type;
      DeviceIndex index;
    } as_device;
  };

  IValue(Payload p, Tag t, bool i) : payload(p), tag(t), is_intrusive_ptr(i) {}

  Payload payload;
  Tag tag;
  bool is_intrusive_ptr;
  friend struct WeakIValue;
};

struct CAFFE2_API WeakIValue final {
  WeakIValue() : payload{0}, tag(IValue::Tag::None), is_intrusive_ptr(false) {}

  WeakIValue(const WeakIValue& rhs)
      : payload(rhs.payload),
        tag(rhs.tag),
        is_intrusive_ptr(rhs.is_intrusive_ptr) {
    if (is_intrusive_ptr) {
      c10::raw::weak_intrusive_ptr::incref(payload.as_intrusive_ptr);
    }
  }
  WeakIValue(const IValue& rhs)
      : payload(rhs.payload),
        tag(rhs.tag),
        is_intrusive_ptr(rhs.is_intrusive_ptr) {
    if (is_intrusive_ptr) {
      c10::raw::weak_intrusive_ptr::incref(payload.as_intrusive_ptr);
    }
  }
  WeakIValue(WeakIValue&& rhs) noexcept : WeakIValue() {
    swap(rhs);
  }
  ~WeakIValue() {
    if (is_intrusive_ptr) {
      c10::raw::weak_intrusive_ptr::decref(payload.as_intrusive_ptr);
    }
  }
  WeakIValue& operator=(WeakIValue&& rhs) & noexcept {
    WeakIValue(std::move(rhs)).swap(*this); // this also sets rhs to None
    return *this;
  }
  WeakIValue& operator=(WeakIValue const& rhs) & {
    WeakIValue(rhs).swap(*this);
    return *this;
  }
  void swap(WeakIValue& rhs) noexcept {
    std::swap(payload, rhs.payload);
    std::swap(is_intrusive_ptr, rhs.is_intrusive_ptr);
    std::swap(tag, rhs.tag);
  }

  bool isSameIdentity(const WeakIValue& rhs) const {
    return payload.as_int == rhs.payload.as_int && tag == rhs.tag &&
        is_intrusive_ptr == rhs.is_intrusive_ptr;
  }

  IValue lock() const {
    if (!is_intrusive_ptr) {
      return IValue(payload, tag, false);
    }
    auto temp = c10::weak_intrusive_ptr<c10::intrusive_ptr_target>::reclaim(
        payload.as_intrusive_ptr);
    IValue::Payload pl;
    pl.as_intrusive_ptr = temp.lock().release();
    temp.release();
    if (!pl.as_intrusive_ptr) {
      return IValue();
    } else {
      return IValue(pl, tag, true);
    }
  }

  size_t use_count() const noexcept {
    if (!is_intrusive_ptr) {
      return 1;
    }
    auto temp = c10::weak_intrusive_ptr<c10::intrusive_ptr_target>::reclaim(
        payload.as_intrusive_ptr);
    size_t result = temp.use_count();
    temp.release();
    return result;
  }

  size_t weak_use_count() const noexcept {
    if (!is_intrusive_ptr) {
      return 1;
    }
    auto temp = c10::weak_intrusive_ptr<c10::intrusive_ptr_target>::reclaim(
        payload.as_intrusive_ptr);
    size_t result = temp.weak_use_count();
    temp.release();
    return result;
  }
  size_t hash() const {
    return payload.as_int;
  }

 private:
  IValue::Payload payload;
  IValue::Tag tag;
  bool is_intrusive_ptr;
};

// An owning pointer to a type. When the type is class type, it requires a pair
// of shared_ptrs to the class type and its owning CU, so that the class type is
// guaranteed to stay alive as long as we hold this object.
struct TORCH_API StrongTypePtr {
  StrongTypePtr(
      std::shared_ptr<torch::jit::CompilationUnit> cu,
      std::shared_ptr<Type> type);

  std::shared_ptr<torch::jit::CompilationUnit> cu_;
  std::shared_ptr<Type> type_;
};

TORCH_API ska::flat_hash_map<std::type_index, c10::ClassTypePtr>&
getCustomClassTypeMap();

template <typename T>
c10::ClassTypePtr getCustomClassTypeImpl() {
  auto& tmap = c10::getCustomClassTypeMap();
  auto res = tmap.find(std::type_index(typeid(T)));
  if (res == tmap.end()) {
    throw c10::Error("Can't find class id in custom class type map", "");
  }
  return res->second;
}

template <typename T>
const c10::ClassTypePtr& getCustomClassType() {
  // Classes are never unregistered from getCustomClassTypeMap and the
  // hash lookup can be a hot path, so just cache.
  // For the same reason, it's fine If this ends up getting duplicated across
  // DSO boundaries for whatever reason.
  static c10::ClassTypePtr cache = getCustomClassTypeImpl<T>();
  return cache;
}

TORCH_API std::unordered_map<std::string, std::function<PyObject*(void*)>>&
getClassConverter();
} // namespace c10

#include <ATen/core/ivalue_inl.h>
