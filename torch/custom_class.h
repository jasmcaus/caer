#pragma once

#include <ATen/core/stack.h>
#include <ATen/core/builtin_function.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/op_registration/infer_schema.h>
#include <ATen/core/stack.h>
#include <c10/util/C++17.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <c10/util/TypeTraits.h>
#include <torch/library.h>
#include <torch/custom_class_detail.h>
#include <iostream>
#include <sstream>

namespace torch {

/// This function is used in conjunction with `class_::def()` to register
/// a constructor for a given C++ class type. For example,
/// `torch::init<int, std::string>()` would register a two-argument constructor
/// taking an `int` and a `std::string` as argument.
template <class... Types>
detail::types<void, Types...> init() {
  return detail::types<void, Types...>{};
}

template <typename Func, typename... ParameterTypeList>
struct InitLambda {
  Func f;
};

template <typename Func>
decltype(auto) init(Func&& f) {
  using InitTraits =
      c10::guts::infer_function_traits_t<std::decay_t<Func>>;
  using ParameterTypeList = typename InitTraits::parameter_types;

  InitLambda<Func, ParameterTypeList> init{std::forward<Func>(f)};
  return init;
}

/// Entry point for custom C++ class registration. To register a C++ class
/// in PyTorch, instantiate `torch::class_` with the desired class as the
/// template parameter. Typically, this instantiation should be done in
/// the initialization of a global variable, so that the class will be
/// made available on dynamic library loading without any additional API
/// calls needed. For example, to register a class named Foo, you might
/// create a global variable like so:
///
///     static auto register_foo = torch::class_<Foo>("myclasses", "Foo")
///       .def("myMethod", &Foo::myMethod)
///       .def("lambdaMethod", [](const c10::intrusive_ptr<Foo>& self) {
///         // Do something with `self`
///       });
///
/// In addition to registering the class, this registration also chains
/// `def()` calls to register methods. `myMethod()` is registered with
/// a pointer to the Foo class's `myMethod()` method. `lambdaMethod()`
/// is registered with a C++ lambda expression.
template <class CurClass>
class class_ {
  static_assert(std::is_base_of<CustomClassHolder, CurClass>::value,
    "torch::class_<T> requires T to inherit from CustomClassHolder");

 public:
  /// This constructor actually registers the class type.
  /// String argument `namespaceName` is an identifier for the
  /// namespace you would like this class to appear in.
  /// String argument `className` is the name you would like to
  /// see this class exposed as in Python and TorchScript. For example, if
  /// you pass `foo` as the namespace name and `Bar` as the className, the
  /// class will appear as `torch.classes.foo.Bar` in Python and TorchScript
  explicit class_(const std::string& namespaceName, const std::string& className, std::string doc_string = "") {
    detail::checkValidIdent(namespaceName, "Namespace name");
    detail::checkValidIdent(className, "Class name");
    qualClassName = std::string("__torch__.torch.classes.") + namespaceName + "." + className;

    classTypePtr = at::ClassType::create(
        c10::QualifiedName(qualClassName),
        std::weak_ptr<jit::CompilationUnit>(),
        /*is_module=*/false,
        std::move(doc_string));
    classTypePtr->addAttribute("capsule", at::CapsuleType::get());

    c10::getCustomClassTypeMap().insert(
        {std::type_index(typeid(c10::intrusive_ptr<CurClass>)), classTypePtr});
    c10::getCustomClassTypeMap().insert(
        {std::type_index(typeid(c10::tagged_capsule<CurClass>)), classTypePtr});

    registerCustomClass(classTypePtr);
  }

  /// def() can be used in conjunction with `torch::init()` to register
  /// a constructor for a given C++ class type. For example, passing
  /// `torch::init<int, std::string>()` would register a two-argument constructor
  /// taking an `int` and a `std::string` as argument.
  template <typename... Types>
  class_& def(detail::types<void, Types...>, std::string doc_string = "") { // Used in combination with
                                               // torch::init<...>()
    auto func = [](c10::tagged_capsule<CurClass> self, Types... args) {
      auto classObj = c10::make_intrusive<CurClass>(args...);
      auto object = self.ivalue.toObject();
      object->setSlot(0, c10::IValue::make_capsule(std::move(classObj)));
    };

    defineMethod("__init__", std::move(func), std::move(doc_string));
    return *this;
  }

  // Used in combination with torch::init([]lambda(){......})
  template <typename Func, typename... ParameterTypes>
  class_& def(
      InitLambda<Func, c10::guts::typelist::typelist<ParameterTypes...>> init,
      std::string doc_string = "") {
    auto init_lambda_wrapper = [func = std::move(init.f)](
                                   c10::tagged_capsule<CurClass> self,
                                   ParameterTypes... arg) {
      c10::intrusive_ptr<CurClass> classObj =
          at::guts::invoke(func, std::forward<ParameterTypes>(arg)...);
      auto object = self.ivalue.toObject();
      object->setSlot(0, c10::IValue::make_capsule(classObj));
    };
    defineMethod("__init__", std::move(init_lambda_wrapper), std::move(doc_string));

    return *this;
  }

  /// This is the normal method registration API. `name` is the name that
  /// the method will be made accessible by in Python and TorchScript.
  /// `f` is a callable object that defines the method. Typically `f`
  /// will either be a pointer to a method on `CurClass`, or a lambda
  /// expression that takes a `c10::intrusive_ptr<CurClass>` as the first
  /// argument (emulating a `this` argument in a C++ method.)
  ///
  /// Examples:
  ///
  ///     // Exposes method `foo` on C++ class `Foo` as `call_foo()` in
  ///     // Python and TorchScript
  ///     .def("call_foo", &Foo::foo)
  ///
  ///     // Exposes the given lambda expression as method `call_lambda()`
  ///     // in Python and TorchScript.
  ///     .def("call_lambda", [](const c10::intrusive_ptr<Foo>& self) {
  ///       // do something
  ///     })
  template <typename Func>
  class_& def(std::string name, Func f, std::string doc_string = "") {
    auto wrapped_f = detail::wrap_func<CurClass, Func>(std::move(f));
    defineMethod(std::move(name), std::move(wrapped_f), std::move(doc_string));
    return *this;
  }

  /// This is an unsafe method registration API added for adding custom JIT backend support via custom
  /// C++ classes. It is not for general purpose use.
  class_& _def_unboxed(std::string name, std::function<void(jit::Stack&)> func, c10::FunctionSchema schema, std::string doc_string = "") {
    auto qualMethodName = qualClassName + "." + name;
    auto method = std::make_unique<jit::BuiltinOpFunction>(
        qualMethodName, std::move(schema), std::move(func), std::move(doc_string));
    classTypePtr->addMethod(method.get());
    registerCustomClassMethod(std::move(method));
    return *this;
  }

  /// def_pickle() is used to define exactly what state gets serialized
  /// or deserialized for a given instance of a custom C++ class in
  /// Python or TorchScript. This protocol is equivalent to the Pickle
  /// concept of `__getstate__` and `__setstate__` from Python
  /// (https://docs.python.org/2/library/pickle.html#object.__getstate__)
  ///
  /// Currently, both the `get_state` and `set_state` callables must be
  /// C++ lambda expressions. They should have the following signatures,
  /// where `CurClass` is the class you're registering and `T1` is some object
  /// that encapsulates the state of the object.
  ///
  ///     __getstate__(intrusive_ptr<CurClass>) -> T1
  ///     __setstate__(T2) -> intrusive_ptr<CurClass>
  ///
  /// `T1` must be an object that is convertable to IValue by the same rules
  /// for custom op/method registration.
  ///
  /// For the common case, T1 == T2. T1 can also be a subtype of T2. An
  /// example where it makes sense for T1 and T2 to differ is if __setstate__
  /// handles legacy formats in a backwards compatible way.
  ///
  /// Example:
  ///
  ///     .def_pickle(
  ///         // __getstate__
  ///         [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
  ///           return self->stack_;
  ///         },
  ///         [](std::vector<std::string> state) { // __setstate__
  ///            return c10::make_intrusive<MyStackClass<std::string>>(
  ///               std::vector<std::string>{"i", "was", "deserialized"});
  ///         })
  template <typename GetStateFn, typename SetStateFn>
  class_& def_pickle(GetStateFn&& get_state, SetStateFn&& set_state) {
    static_assert(
        c10::guts::is_stateless_lambda<std::decay_t<GetStateFn>>::value &&
            c10::guts::is_stateless_lambda<std::decay_t<SetStateFn>>::value,
        "def_pickle() currently only supports lambdas as "
        "__getstate__ and __setstate__ arguments.");
    def("__getstate__", std::forward<GetStateFn>(get_state));

    // __setstate__ needs to be registered with some custom handling:
    // We need to wrap the invocation of of the user-provided function
    // such that we take the return value (i.e. c10::intrusive_ptr<CurrClass>)
    // and assign it to the `capsule` attribute.
    using SetStateTraits =
        c10::guts::infer_function_traits_t<std::decay_t<SetStateFn>>;
    using SetStateArg = typename c10::guts::typelist::head_t<
        typename SetStateTraits::parameter_types>;
    auto setstate_wrapper = [set_state = std::move(set_state)](
                                c10::tagged_capsule<CurClass> self,
                                SetStateArg&& arg) {
      c10::intrusive_ptr<CurClass> classObj =
          at::guts::invoke(set_state, std::forward<SetStateArg>(arg));
      auto object = self.ivalue.toObject();
      object->setSlot(0, c10::IValue::make_capsule(classObj));
    };
    defineMethod(
        "__setstate__",
        detail::wrap_func<CurClass, decltype(setstate_wrapper)>(
            std::move(setstate_wrapper)));

    // type validation
    auto getstate_schema = classTypePtr->getMethod("__getstate__").getSchema();
    auto format_getstate_schema = [&getstate_schema]() {
      std::stringstream ss;
      ss << getstate_schema;
      return ss.str();
    };
    TORCH_CHECK(
        getstate_schema.arguments().size() == 1,
        "__getstate__ should take exactly one argument: self. Got: ",
        format_getstate_schema());
    auto first_arg_type = getstate_schema.arguments().at(0).type();
    TORCH_CHECK(
        *first_arg_type == *classTypePtr,
        "self argument of __getstate__ must be the custom class type. Got ",
        first_arg_type->repr_str());
    TORCH_CHECK(
        getstate_schema.returns().size() == 1,
        "__getstate__ should return exactly one value for serialization. Got: ",
        format_getstate_schema());

    auto ser_type = getstate_schema.returns().at(0).type();
    auto setstate_schema = classTypePtr->getMethod("__setstate__").getSchema();
    auto arg_type = setstate_schema.arguments().at(1).type();
    TORCH_CHECK(
        ser_type->isSubtypeOf(arg_type),
        "__getstate__'s return type should be a subtype of "
        "input argument of __setstate__. Got ",
        ser_type->repr_str(),
        " but expected ",
        arg_type->repr_str());

    return *this;
  }

 private:
  template <typename Func>
  void defineMethod(std::string name, Func func, std::string doc_string = "") {
    auto qualMethodName = qualClassName + "." + name;
    auto schema = c10::inferFunctionSchemaSingleReturn<Func>(std::move(name), "");

    auto wrapped_func = [func = std::move(func)](jit::Stack& stack) mutable -> void {
      // TODO: we need to figure out how to profile calls to custom functions
      // like this! Currently can't do it because the profiler stuff is in
      // libtorch and not ATen
      using RetType =
          typename c10::guts::infer_function_traits_t<Func>::return_type;
      detail::BoxedProxy<RetType, Func>()(stack, func);
    };
    auto method = std::make_unique<jit::BuiltinOpFunction>(
        qualMethodName, std::move(schema), std::move(wrapped_func), std::move(doc_string));

    // Register the method here to keep the Method alive.
    // ClassTypes do not hold ownership of their methods (normally it
    // those are held by the CompilationUnit), so we need a proxy for
    // that behavior here.
    classTypePtr->addMethod(method.get());
    registerCustomClassMethod(std::move(method));
  }

  std::string qualClassName;
  at::ClassTypePtr classTypePtr;
};

/// make_custom_class() is a convenient way to create an instance of a registered
/// custom class and wrap it in an IValue, for example when you want to pass the
/// object to TorchScript. Its syntax is equivalent to APIs like `std::make_shared<>`
/// or `c10::make_intrusive<>`.
///
/// For example, if you have a custom C++ class that can be constructed from an `int`
/// and `std::string`, you might use this API like so:
///
///     IValue custom_class_iv = torch::make_custom_class<MyClass>(3, "foobarbaz");
template <typename CurClass, typename... CtorArgs>
c10::IValue make_custom_class(CtorArgs&&... args) {
  auto userClassInstance = c10::make_intrusive<CurClass>(std::forward<CtorArgs>(args)...);
  return c10::IValue(std::move(userClassInstance));
}

// jit namespace for backward-compatibility
// We previously defined everything in torch::jit but moved it out to
// better reflect that these features are not limited only to TorchScript
namespace jit {

using ::torch::getCustomClass;
using ::torch::isCustomClass;
using ::torch::init;
using ::torch::class_;

} // namespace jit

template <class CurClass>
inline class_<CurClass> Library::class_(const std::string& className) {
  TORCH_CHECK(kind_ == DEF || kind_ == FRAGMENT,
    "class_(\"", className, "\"): Cannot define a class inside of a TORCH_LIBRARY_IMPL block.  "
    "All class_()s should be placed in the (unique) TORCH_LIBRARY block for their namespace.  "
    "(Error occurred at ", file_, ":", line_, ")");
  TORCH_INTERNAL_ASSERT(ns_.has_value(), file_, ":", line_);
  return torch::class_<CurClass>(*ns_, className);
}

}
