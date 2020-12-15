#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeIndex.h>
#include <gtest/gtest.h>

using c10::string_view;
using c10::util::get_fully_qualified_type_name;
using c10::util::get_type_index;

namespace {

static_assert(get_type_index<int>() == get_type_index<int>(), "");
static_assert(get_type_index<float>() == get_type_index<float>(), "");
static_assert(get_type_index<int>() != get_type_index<float>(), "");
static_assert(
    get_type_index<int(double, double)>() ==
        get_type_index<int(double, double)>(),
    "");
static_assert(
    get_type_index<int(double, double)>() != get_type_index<int(double)>(),
    "");
static_assert(
    get_type_index<int(double, double)>() ==
        get_type_index<int (*)(double, double)>(),
    "");
static_assert(
    get_type_index<std::function<int(double, double)>>() ==
        get_type_index<std::function<int(double, double)>>(),
    "");
static_assert(
    get_type_index<std::function<int(double, double)>>() !=
        get_type_index<std::function<int(double)>>(),
    "");

static_assert(get_type_index<int>() == get_type_index<int&>(), "");
static_assert(get_type_index<int>() == get_type_index<int&&>(), "");
static_assert(get_type_index<int>() == get_type_index<const int&>(), "");
static_assert(get_type_index<int>() == get_type_index<const int>(), "");
static_assert(get_type_index<const int>() == get_type_index<int&>(), "");
static_assert(get_type_index<int>() != get_type_index<int*>(), "");
static_assert(get_type_index<int*>() != get_type_index<int**>(), "");
static_assert(
    get_type_index<int(double&, double)>() !=
        get_type_index<int(double, double)>(),
    "");

struct Dummy final {};
struct Functor final {
  int64_t operator()(uint32_t, Dummy&&, const Dummy&) const;
};
static_assert(
    get_type_index<int64_t(uint32_t, Dummy&&, const Dummy&)>() ==
        get_type_index<
            c10::guts::infer_function_traits_t<Functor>::func_type>(),
    "");

namespace test_top_level_name {
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos != get_fully_qualified_type_name<Dummy>().find("Dummy"),
    "");
#endif
TEST(TypeIndex, TopLevelName) {
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<Dummy>().find("Dummy")
    );
}
}

namespace test_nested_name {
struct Dummy final {};

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Dummy>().find("test_nested_name::Dummy"),
    "");
#endif
TEST(TypeIndex, NestedName) {
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<Dummy>().find("test_nested_name::Dummy")
    );
}
} // namespace test_nested_name

namespace test_type_template_parameter {
template <class T>
struct Outer final {};
struct Inner final {};

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Outer<Inner>>().find(
            "test_type_template_parameter::Outer"),
    "");
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Outer<Inner>>().find(
            "test_type_template_parameter::Inner"),
    "");
#endif
TEST(TypeIndex, TypeTemplateParameter) {
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<Outer<Inner>>().find(
            "test_type_template_parameter::Outer")
    );
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<Outer<Inner>>().find(
            "test_type_template_parameter::Inner")
    );
}
} // namespace test_type_template_parameter

namespace test_nontype_template_parameter {
template <size_t N>
struct Class final {};

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Class<38474355>>().find("38474355"),
    "");
#endif
TEST(TypeIndex, NonTypeTemplateParameter) {
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<Class<38474355>>().find("38474355")
    );
}
} // namespace test_nontype_template_parameter

namespace test_type_computations_are_resolved {
template <class T>
struct Type final {
  using type = const T*;
};

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<typename Type<int>::type>().find("int"),
    "");
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<typename Type<int>::type>().find("*"),
    "");

// but with remove_pointer applied, there is no '*' in the type name anymore
static_assert(
    string_view::npos ==
        get_fully_qualified_type_name<
            typename std::remove_pointer<typename Type<int>::type>::type>()
            .find("*"),
    "");
#endif
TEST(TypeIndex, TypeComputationsAreResolved) {
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<typename Type<int>::type>().find("int")
    );
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<typename Type<int>::type>().find("*")
    );
    // but with remove_pointer applied, there is no '*' in the type name anymore
    EXPECT_EQ(
        string_view::npos,
        get_fully_qualified_type_name<
            typename std::remove_pointer<typename Type<int>::type>::type>()
            .find("*")
    );
}

struct Functor final {
  std::string operator()(int64_t a, const Type<int>& b) const;
};
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    get_fully_qualified_type_name<std::string(int64_t, const Type<int>&)>() ==
        get_fully_qualified_type_name<
            typename c10::guts::infer_function_traits_t<Functor>::func_type>(),
    "");
#endif
TEST(TypeIndex, FunctionTypeComputationsAreResolved) {
    EXPECT_EQ(
        get_fully_qualified_type_name<std::string(int64_t, const Type<int>&)>(),
        get_fully_qualified_type_name<
            typename c10::guts::infer_function_traits_t<Functor>::func_type>()
    );
}
} // namespace test_type_computations_are_resolved

namespace test_function_arguments_and_returns {
class Dummy final {};

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Dummy(int)>().find(
            "test_function_arguments_and_returns::Dummy"),
    "");
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<void(Dummy)>().find(
            "test_function_arguments_and_returns::Dummy"),
    "");
#endif
TEST(TypeIndex, FunctionArgumentsAndReturns) {
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<Dummy(int)>().find(
            "test_function_arguments_and_returns::Dummy")
    );
    EXPECT_NE(
        string_view::npos,
        get_fully_qualified_type_name<void(Dummy)>().find(
            "test_function_arguments_and_returns::Dummy")
    );
}
} // namespace test_function_arguments_and_returns
} // namespace
