#pragma once

#include <ATen/core/function_schema.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/either.h>
#include <c10/util/Optional.h>
#include <c10/core/DispatchKey.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/DispatchKeyExtractor.h>

#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/dispatch/CppSignature.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>

#include <list>
#include <array>

namespace c10 {

class Dispatcher;

namespace impl {

// This data structure represents a kernel that was registered to us from a
// user.  Unlike KernelFunction, AnnotatedKernel contains some extra metadata
// about the kernel that isn't necessary for actual dispatching (this is why
// we don't put AnnotatedKernel in the actual DispatchTable), but is useful for
// giving good error messages.
struct AnnotatedKernel final {
  AnnotatedKernel(KernelFunction k, std::unique_ptr<FunctionSchema> s, std::string d)
    : kernel(std::move(k))
    , inferred_function_schema(std::move(s))
    , debug(std::move(d))
    {}
  AnnotatedKernel() {}
  KernelFunction kernel;
  std::unique_ptr<FunctionSchema> inferred_function_schema;
  // A little debug string to help us identify the kernel in question.
  // Most importantly it records the TORCH_LIBRARY block that did the
  // registration.
  std::string debug;
};

// This data structure represents operator schema, with metadata specifying
// where the registration of this schema occurred
struct AnnotatedSchema final {
  AnnotatedSchema(FunctionSchema s, std::string d)
    : schema(std::move(s))
    , debug(std::move(d))
    {}
  FunctionSchema schema;
  std::string debug;
};

// Internal data structure that records information about a specific operator.
// It's not part of the public API; typically, users will interact with
// OperatorHandle instead.
//
// Concurrent writes to OperatorEntry are protected by the GLOBAL Dispatcher
// lock (this is important because some methods in OperatorEntry access
// dispatcher state)
class CAFFE2_API OperatorEntry final {
public:
  explicit OperatorEntry(OperatorName&& operator_name);

  OperatorEntry(const OperatorEntry&) = delete;
  OperatorEntry(OperatorEntry&&) noexcept = delete;
  OperatorEntry& operator=(const OperatorEntry&) = delete;
  OperatorEntry& operator=(OperatorEntry&&) noexcept = delete;

  const FunctionSchema& schema() const {
    TORCH_INTERNAL_ASSERT(schema_.has_value(), "Tried to access the schema for ", name_, " which doesn't have a schema registered yet");
    return schema_->schema;
  }
  const std::string& debug() const {
    TORCH_INTERNAL_ASSERT(schema_.has_value());
    return schema_->debug;
  }
  bool hasSchema() const {
    return schema_.has_value();
  }

  bool isObserved() const {
    return is_observed_;
  }

  // We may allocate an OperatorEntry for an operator even when we don't
  // have a schema.  When we receive the schema registration, we post
  // facto register a schema.
  //
  // NB: registerSchema/deregisterSchema are not idempotent; if you
  // attempt to register a schema when one is already present or vice
  // versa that is an error.  (Refcounting for the registrations is
  // handled in the OperatorHandle in Dispatcher)
  void registerSchema(FunctionSchema&&, std::string&& debug);
  void deregisterSchema();

  const OperatorName& operator_name() const {
    return name_;
  }

  // Why are kernels and fallback asymmetric?  It has to do with ownership.
  // Kernels and the computed dispatch tables for them are canonically
  // owned by OperatorEntry, but backend fallbacks are specified once
  // and apply for all operators, so they should be owned by Dispatcher.
  // However, the registration of a backend fallback affects the
  // state of the computed dispatch table, so when a backend fallback
  // is updated, we need to update the operator tables too.  Thus,
  // registerKernel is the mechanism by which we give kernels to
  // operator entry to own (and update dispatch table), but we only
  // need a non-owning mechanism to update fallback.

  // Precondition: Dispatcher::mutex_ is held
  // Postcondition: caller is responsible for disposing of the kernel
  std::list<AnnotatedKernel>::iterator registerKernel(
    const Dispatcher& dispatcher,
    c10::optional<DispatchKey> dispatch_key,
    KernelFunction kernel,
    c10::optional<CppSignature> cpp_signature,
    std::unique_ptr<FunctionSchema> inferred_function_schema,
    std::string debug
  );

  // Precondition: Dispatcher::mutex_ is held
  void deregisterKernel_(
    const Dispatcher& dispatcher,
    c10::optional<DispatchKey> dispatch_key,
    std::list<AnnotatedKernel>::iterator kernel
  );

  // Precondition: Dispatcher::mutex_ is held
  void updateFallback(
    const Dispatcher& dispatcher,
    DispatchKey dispatch_key
  );

  // Precondition: Dispatcher::mutex_ is held
  void updateSchemaAliasAnalysis(AliasAnalysisKind a) {
    TORCH_INTERNAL_ASSERT(schema_.has_value());
    schema_->schema.setAliasAnalysis(a);
  }

  std::string dumpComputedTable() const;
  std::string dumpState() const;
  void checkInvariants() const;

  const DispatchKeyExtractor& dispatchKeyExtractor() const { return dispatchKeyExtractor_; }

  // This function is a temporary hack that allows generated_unboxing_wrappers.cpp to register its codegen'ed
  // unboxing wrapper for aten operators. We still need those for some operators because not all work
  // with the templated unboxing logic yet.
  // TODO Delete setManuallyBoxedKernel_ once all operators work with the templated boxing logic
  void setManuallyBoxedKernel_(const c10::Dispatcher& dispatcher, KernelFunction::InternalBoxedKernelFunction* func);

  // Asserts that the given FuncType is correct for calling this operator in an unboxed way.
  template<class FuncType>
  void assertSignatureIsCorrect() {
    TORCH_CHECK(!cpp_signature_.has_value() || (CppSignature::make<FuncType>() == cpp_signature_->signature),
        "\nTried to access or call an operator with a wrong signature.\n",
        "  operator: ", (schema_.has_value() ? toString(schema_->schema) : toString(name_)), "\n",
        "    ", (schema_.has_value() ? schema_->debug : "unknown debug info"), "\n",
        "  correct signature:  ", cpp_signature_->signature.name(), "\n",
        "    ", cpp_signature_->debug, "\n",
        "  accessed/called as: ", CppSignature::make<FuncType>().name(), "\n",
        "This likely happened in a call to OperatorHandle::typed<Return (Args...)>(). ",
        "Please make sure that the function signature matches the signature in the operator registration call."
    );
  }

  [[noreturn]] void reportError(DispatchKey dispatchKey) const;

  const KernelFunction& lookup(DispatchKey k) const {
    const auto& kernel = dispatchTable_[static_cast<uint8_t>(k)];
    if (C10_UNLIKELY(!kernel.isValid())) {
      reportError(k);
    }
    return kernel;
  }

  std::string listAllDispatchKeys() const;

private:

  OperatorName name_;
  c10::optional<AnnotatedSchema> schema_;

  std::array<KernelFunction, static_cast<uint8_t>(DispatchKey::NumDispatchKeys)> dispatchTable_;
  DispatchKeyExtractor dispatchKeyExtractor_;

  // This manuallyBoxedKernel_ member is a temporary hack that allows generated_unboxing_wrappers.cpp to register its codegen'ed
  // unboxing wrapper for aten operators. We still need those for some operators because not all work
  // with the templated unboxing logic yet.
  // TODO Delete manuallyBoxedKernel_ once all operators work with the templated boxing logic
  c10::optional<KernelFunction::InternalBoxedKernelFunction*> manuallyBoxedKernel_;

  // kernels_ stores all registered kernels for the corresponding dispatch key
  // and catchAllKernels_ stores the catch-all kernels.
  // If an operator library gets loaded that overwrites an already existing kernel,
  // both kernels will be in that list but only the newer one will be in
  // dispatchTable. If any of the kernels go away (say the library gets
  // unloaded), we remove the kernel from this list and update the
  // dispatchTable if necessary.
  // Kernels in the list are ordered by registration time descendingly,
  // newer registrations are before older registrations.
  // We do not combine dispatchTable and kernels into one hash map because
  // kernels is a larger data structure and accessed quite infrequently
  // while dispatchTable is accessed often and should be kept small to fit
  // into CPU caches.
  // Invariants:
  //  - dispatchTable[dispatch_key] == kernels_[dispatch_key].front()
  //  - dispatchTable[dispatch_key] does not exist if and only if
  //    kernels_[dispatch_key] does not exist
  //  - If kernels_[dispatch_key] exists, then it has elements.
  //    It is never an empty list.
  //
  // Why do we do that?
  // -----
  // We mostly do this to enable Jupyter notebooks where a cell registering
  // a kernel could be executed multiple times and the later execution
  // should overwrite the earlier one. Note that this still fails when the
  // function schema changed between the executions, but it works as long
  // as the function schema didn't change. A better solution would be to
  // unload the old extension library from the Jupyter cell when the cell is
  // re-executed and then only allow one kernel here, i.e. error if a kernel
  // is already registered, but that's a lot of effort to implement and
  // currently not high-pri.
  ska::flat_hash_map<DispatchKey, std::list<AnnotatedKernel>> kernels_;

  AnnotatedKernel missingKernel_;
  static const AnnotatedKernel ambiguousAutogradOtherKernel_;

  // cpp_signature_ stores function signature if any of
  // the kernels was created in a way that allowed us to know the function
  // signature (i.e. by supplying an unboxed C++ kernel function).
  // If this is set, it will be used to check that future kernel
  // registrations match and it will be used in unboxed function calls
  // to verify their arguments against the known function signature.
  struct CppSignatureWithDebug {
    CppSignature signature;
    std::string debug;
    c10::optional<DispatchKey> dispatch_key;
  };
  c10::optional<CppSignatureWithDebug> cpp_signature_;

  // Whether this operator needs to be observed with RecordFunction
  const bool is_observed_;

  const KernelFunction& computeDispatchTableEntry(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) const;
  std::pair<const AnnotatedKernel&, const char*> computeDispatchTableEntryWithDebug(
    const c10::Dispatcher& dispatcher, DispatchKey dispatch_key
  ) const;
  // This function re-establishes the invariant that dispatchTable
  // contains the front element from the kernels list for a given runtime dispatch key.
  void updateDispatchTableEntry_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key);
  // Like above, but also handles alias dispatch keys.
  void updateDispatchTable_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key);
  // Like above, but for ALL entries in the dispatch table.
  void updateDispatchTableFull_(const c10::Dispatcher& dispatcher);

  // Returns true if kernel_ has entry for any key in ks.
  bool hasKernelForAnyDispatchKey(DispatchKeySet ks) const;
  // Retrieves a pointer to AnnotatedKernel at kernels_.at(dispatch_key).front().
  c10::optional<const AnnotatedKernel*> getKernelForDispatchKey(DispatchKey dispatch_key) const;
};

} // namespace impl
} // namespace c10
