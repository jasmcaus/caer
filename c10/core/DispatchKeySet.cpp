#include <c10/core/DispatchKeySet.h>

namespace c10 {

// backend_dispatch_keyset should include all runtime backend keys.
// Alias key DispatchKey::DefaultBackend maps to backend_dispatch_keyset
constexpr DispatchKeySet backend_dispatch_keyset = autogradother_backends | DispatchKeySet({
  DispatchKey::CPU,
  DispatchKey::CUDA,
  DispatchKey::XLA,
  DispatchKey::NestedTensor,
  DispatchKey::PrivateUse1,
  DispatchKey::PrivateUse2,
  DispatchKey::PrivateUse3,
});

bool isBackendDispatchKey(DispatchKey t) {
  return t != DispatchKey::Undefined && backend_dispatch_keyset.has(t);
}

// math_dispatch_keyset contains all keys in backend_dispatch_keyset and autograd_dispatch_keyset
// Alias key DispatchKey::Math maps to math_dispatch_keyset.
constexpr DispatchKeySet math_dispatch_keyset = backend_dispatch_keyset | autograd_dispatch_keyset;

DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t) {
  TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
  switch (t) {
    case DispatchKey::Autograd:
      return autograd_dispatch_keyset;
    case DispatchKey::Math:
      return math_dispatch_keyset;
    case DispatchKey::DefaultBackend:
      return backend_dispatch_keyset;
    default:
      return DispatchKeySet(t);
  }
}

// for a given autograd key, return the (guaranteed nonempty) set of associated backend keys.
// for a non-autograd key, return the empty keyset.
DispatchKeySet getBackendKeySetFromAutograd(DispatchKey t) {
  switch (t) {
    case DispatchKey::AutogradCPU:
      return DispatchKeySet(DispatchKey::CPU);
    case DispatchKey::AutogradCUDA:
      return DispatchKeySet(DispatchKey::CUDA);
    case DispatchKey::AutogradXLA:
      return DispatchKeySet(DispatchKey::XLA);
    case DispatchKey::AutogradNestedTensor:
      return DispatchKeySet(DispatchKey::NestedTensor);
    case DispatchKey::AutogradPrivateUse1:
      return DispatchKeySet(DispatchKey::PrivateUse1);
    case DispatchKey::AutogradPrivateUse2:
      return DispatchKeySet(DispatchKey::PrivateUse2);
    case DispatchKey::AutogradPrivateUse3:
      return DispatchKeySet(DispatchKey::PrivateUse3);
    case DispatchKey::AutogradOther:
      return autogradother_backends;
    default:
      return DispatchKeySet();
  }
}

bool isIncludedInAlias(DispatchKey k, DispatchKey alias) {
  return k != DispatchKey::Undefined && getRuntimeDispatchKeySet(alias).has(k);
}

std::string toString(DispatchKeySet ts) {
  std::stringstream ss;
  ss << ts;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, DispatchKeySet ts) {
  if (ts.empty()) {
    os << "DispatchKeySet()";
    return os;
  }
  os << "DispatchKeySet(";
  DispatchKey tid;
  bool first = true;
  while ((tid = ts.highestPriorityTypeId()) != DispatchKey::Undefined) {
    if (!first) {
      os << ", ";
    }
    os << tid;
    ts = ts.remove(tid);
    first = false;
  }
  os << ")";
  return os;
}

}
