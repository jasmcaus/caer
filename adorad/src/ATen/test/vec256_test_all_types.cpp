#include <ATen/test/vec256_test_all_types.h>
namespace {
#if GTEST_HAS_TYPED_TEST
    template <typename T>
    class Memory : public ::testing::Test {};
    template <typename T>
    class Arithmetics : public ::testing::Test {};
    template <typename T>
    class Comparison : public ::testing::Test {};
    template <typename T>
    class Bitwise : public ::testing::Test {};
    template <typename T>
    class MinMax : public ::testing::Test {};
    template <typename T>
    class Interleave : public ::testing::Test {};
    template <typename T>
    class SignManipulation : public ::testing::Test {};
    template <typename T>
    class Rounding : public ::testing::Test {};
    template <typename T>
    class SqrtAndReciprocal : public ::testing::Test {};
    template <typename T>
    class SqrtAndReciprocalReal : public ::testing::Test {};
    template <typename T>
    class FractionAndRemainderReal : public ::testing::Test {};
    template <typename T>
    class Trigonometric : public ::testing::Test {};
    template <typename T>
    class ErrorFunctions : public ::testing::Test {};
    template <typename T>
    class Exponents : public ::testing::Test {};
    template <typename T>
    class Hyperbolic : public ::testing::Test {};
    template <typename T>
    class InverseTrigonometric : public ::testing::Test {};
    template <typename T>
    class InverseTrigonometricReal : public ::testing::Test {};
    template <typename T>
    class LGamma : public ::testing::Test {};
    template <typename T>
    class Logarithm : public ::testing::Test {};
    template <typename T>
    class LogarithmReals : public ::testing::Test {};
    template <typename T>
    class Pow : public ::testing::Test {};
    template <typename T>
    class RangeFactories : public ::testing::Test {};
    template <typename T>
    class BitwiseFloatsAdditional : public ::testing::Test {};
    template <typename T>
    class BitwiseFloatsAdditional2 : public ::testing::Test {};
    template <typename T>
    class RealTests : public ::testing::Test {};
    template <typename T>
    class ComplexTests : public ::testing::Test {};
    template <typename T>
    class QuantizationTests : public ::testing::Test {};
    using RealFloatTestedTypes = ::testing::Types<vfloat, vdouble>;
    using FloatTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vcomplexDbl>;
    using ALLTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vlong, vint, vshort, vqint8, vquint8, vqint>;
    using QuantTestedTypes = ::testing::Types<vqint8, vquint8, vqint>;
    using RealFloatIntTestedTypes = ::testing::Types<vfloat, vdouble, vlong, vint, vshort>;
    using FloatIntTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vcomplexDbl, vlong, vint, vshort>;
    using ComplexTypes = ::testing::Types<vcomplex, vcomplexDbl>;
    TYPED_TEST_CASE(Memory, ALLTestedTypes);
    TYPED_TEST_CASE(Arithmetics, FloatIntTestedTypes);
    TYPED_TEST_CASE(Comparison, RealFloatIntTestedTypes);
    TYPED_TEST_CASE(Bitwise, FloatIntTestedTypes);
    TYPED_TEST_CASE(MinMax, RealFloatIntTestedTypes);
    TYPED_TEST_CASE(Interleave, RealFloatIntTestedTypes);
    TYPED_TEST_CASE(SignManipulation, FloatIntTestedTypes);
    TYPED_TEST_CASE(Rounding, RealFloatTestedTypes);
    TYPED_TEST_CASE(SqrtAndReciprocal, FloatTestedTypes);
    TYPED_TEST_CASE(SqrtAndReciprocalReal, RealFloatTestedTypes);
    TYPED_TEST_CASE(FractionAndRemainderReal, RealFloatTestedTypes);
    TYPED_TEST_CASE(Trigonometric, RealFloatTestedTypes);
    TYPED_TEST_CASE(ErrorFunctions, RealFloatTestedTypes);
    TYPED_TEST_CASE(Exponents, RealFloatTestedTypes);
    TYPED_TEST_CASE(Hyperbolic, RealFloatTestedTypes);
    TYPED_TEST_CASE(InverseTrigonometricReal, RealFloatTestedTypes);
    TYPED_TEST_CASE(InverseTrigonometric, FloatTestedTypes);
    TYPED_TEST_CASE(LGamma, RealFloatTestedTypes);
    TYPED_TEST_CASE(Logarithm, FloatTestedTypes);
    TYPED_TEST_CASE(LogarithmReals, RealFloatTestedTypes);
    TYPED_TEST_CASE(Pow, RealFloatTestedTypes);
    TYPED_TEST_CASE(RealTests, RealFloatTestedTypes);
    TYPED_TEST_CASE(RangeFactories, FloatIntTestedTypes);
    TYPED_TEST_CASE(BitwiseFloatsAdditional, RealFloatTestedTypes);
    TYPED_TEST_CASE(BitwiseFloatsAdditional2, FloatTestedTypes);
    TYPED_TEST_CASE(QuantizationTests, QuantTestedTypes);
    TYPED_TEST(Memory, UnAlignedLoadStore) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr size_t b_size = vec::size() * sizeof(VT);
        CACHE_ALIGN unsigned char ref_storage[128 * b_size];
        CACHE_ALIGN unsigned char storage[128 * b_size];
        auto seed = TestSeed();
        ValueGen<unsigned char> generator(seed);
        for (auto& x : ref_storage) {
            x = generator.get();
        }
        // test counted load stores
#if defined(CPU_CAPABILITY_VSX)
        for (int i = 1; i < 2 * vec::size(); i++) {
            vec v = vec::loadu(ref_storage, i);
            v.store(storage);
            size_t count = std::min(i * sizeof(VT), b_size);
            bool cmp = (std::memcmp(ref_storage, storage, count) == 0);
            ASSERT_TRUE(cmp) << "Failure Details:\nTest Seed to reproduce: " << seed
                << "\nCount: " << i;
            if (::testing::Test::HasFailure()) {
                break;
            }
            // clear storage
            std::memset(storage, 0, b_size);
        }
#endif
        // testing unaligned load store
        for (size_t offset = 0; offset < b_size; offset += 1) {
            unsigned char* p1 = ref_storage + offset;
            unsigned char* p2 = storage + offset;
            for (; p1 + b_size <= std::end(ref_storage); p1 += b_size, p2 += b_size) {
                vec v = vec::loadu(p1);
                v.store(p2);
            }
            size_t written = p1 - ref_storage - offset;
            bool cmp = (std::memcmp(ref_storage + offset, storage + offset, written) == 0);
            ASSERT_TRUE(cmp) << "Failure Details:\nTest Seed to reproduce: " << seed
                << "\nMismatch at unaligned offset: " << offset;
            if (::testing::Test::HasFailure()) {
                break;
            }
            // clear storage
            std::memset(storage, 0, sizeof storage);
        }
    }
    TYPED_TEST(SignManipulation, Absolute) {
        using vec = TypeParam;
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        test_unary<vec>(
            NAME_INFO(absolute), RESOLVE_OVERLOAD(local_abs),
            [](vec v) { return v.abs(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, checkRelativeErr),
            RESOLVE_OVERLOAD(filter_int_minimum));
    }
    TYPED_TEST(SignManipulation, Negate) {
        using vec = TypeParam;
        // negate overflows for minimum on int and long
        test_unary<vec>(
            NAME_INFO(negate), std::negate<ValueType<vec>>(),
            [](vec v) { return v.neg(); },
            createDefaultUnaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_int_minimum));
    }
    TYPED_TEST(Rounding, Round) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        UVT case1 = -658.5f;
        UVT exp1 = -658.f;
        UVT case2 = -657.5f;
        UVT exp2 = -658.f;
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-1000, 1000}} })
            .addCustom({ {case1},exp1 })
            .addCustom({ {case2},exp2 })
            .setTrialCount(64000)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(round),
            RESOLVE_OVERLOAD(at::native::round_impl),
            [](vec v) { return v.round(); },
            test_case);
    }
    TYPED_TEST(Rounding, Ceil) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(ceil),
            RESOLVE_OVERLOAD(std::ceil),
            [](vec v) { return v.ceil(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Rounding, Floor) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(floor),
            RESOLVE_OVERLOAD(std::floor),
            [](vec v) { return v.floor(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Rounding, Trunc) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(trunc),
            RESOLVE_OVERLOAD(std::trunc),
            [](vec v) { return v.trunc(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(SqrtAndReciprocal, Sqrt) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(sqrt),
            RESOLVE_OVERLOAD(local_sqrt),
            [](vec v) { return v.sqrt(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(SqrtAndReciprocalReal, RSqrt) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(rsqrt),
            rsqrt<ValueType<vec>>,
            [](vec v) { return v.rsqrt(); },
            createDefaultUnaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_zero));
    }
    TYPED_TEST(SqrtAndReciprocalReal, Reciprocal) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(reciprocal),
            reciprocal<ValueType<vec>>,
            [](vec v) { return v.reciprocal(); },
            createDefaultUnaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_zero));
    }
    TYPED_TEST(FractionAndRemainderReal, Frac) {
      using vec = TypeParam;
      test_unary<vec>(
          NAME_INFO(frac),
          RESOLVE_OVERLOAD(frac),
          [](vec v) { return v.frac(); },
          createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(FractionAndRemainderReal, Fmod) {
      using vec = TypeParam;
      test_binary<vec>(
          NAME_INFO(fmod),
          RESOLVE_OVERLOAD(std::fmod),
          [](vec v0, vec v1) { return v0.fmod(v1); },
          createDefaultBinaryTestCase<vec>(TestSeed()),
          RESOLVE_OVERLOAD(filter_fmod));
    }
    TYPED_TEST(Trigonometric, Sin) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-4096, 4096}}, true, 1.2e-7f})
            .addDomain(CheckWithinDomains<UVT>{ { {-8192, 8192}}, true, 3.0e-7f})
            .setTrialCount(8000)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(sin),
            RESOLVE_OVERLOAD(std::sin),
            [](vec v) { return v.sin(); },
            test_case);
    }
    TYPED_TEST(Trigonometric, Cos) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-4096, 4096}}, true, 1.2e-7f})
            .addDomain(CheckWithinDomains<UVT>{ { {-8192, 8192}}, true, 3.0e-7f})
            .setTrialCount(8000)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(cos),
            RESOLVE_OVERLOAD(std::cos),
            [](vec v) { return v.cos(); },
            test_case);
    }
    TYPED_TEST(Trigonometric, Tan) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(tan),
            RESOLVE_OVERLOAD(std::tan),
            [](vec v) { return v.tan(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Hyperbolic, Tanh) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(tanH),
            RESOLVE_OVERLOAD(std::tanh),
            [](vec v) { return v.tanh(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Hyperbolic, Sinh) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-88, 88}}, true, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(sinh),
            RESOLVE_OVERLOAD(std::sinh),
            [](vec v) { return v.sinh(); },
            test_case);
    }
    TYPED_TEST(Hyperbolic, Cosh) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-88, 88}}, true, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(cosh),
            RESOLVE_OVERLOAD(std::cosh),
            [](vec v) { return v.cosh(); },
            test_case);
    }
    TYPED_TEST(InverseTrigonometric, Asin) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-10, 10}}, checkRelativeErr, getDefaultTolerance<UVT>() })
            .setTrialCount(125536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(asin),
            RESOLVE_OVERLOAD(local_asin),
            [](vec v) { return v.asin(); },
            test_case);
    }
    TYPED_TEST(InverseTrigonometric, ACos) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-10, 10}}, checkRelativeErr, getDefaultTolerance<UVT>() })
            .setTrialCount(125536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(acos),
            RESOLVE_OVERLOAD(local_acos),
            [](vec v) { return v.acos(); },
            test_case);
    }
    TYPED_TEST(InverseTrigonometric, ATan) {
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-100, 100}}, checkRelativeErr, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(atan),
            RESOLVE_OVERLOAD(std::atan),
            [](vec v) { return v.atan(); },
            test_case,
            RESOLVE_OVERLOAD(filter_zero));
    }
    TYPED_TEST(Logarithm, Log) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(log),
            RESOLVE_OVERLOAD(std::log),
            [](const vec& v) { return v.log(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(LogarithmReals, Log2) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(log2),
            RESOLVE_OVERLOAD(local_log2),
            [](const vec& v) { return v.log2(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Logarithm, Log10) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(log10),
            RESOLVE_OVERLOAD(std::log10),
            [](const vec& v) { return v.log10(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(LogarithmReals, Log1p) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-1, 1000}}, true, getDefaultTolerance<UVT>()})
            .addDomain(CheckWithinDomains<UVT>{ { {1000, 1.e+30}}, true, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(log1p),
            RESOLVE_OVERLOAD(std::log1p),
            [](const vec& v) { return v.log1p(); },
            test_case);
    }
    TYPED_TEST(Exponents, Exp) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(exp),
            RESOLVE_OVERLOAD(std::exp),
            [](const vec& v) { return v.exp(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Exponents, Expm1) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(expm1),
            RESOLVE_OVERLOAD(std::expm1),
            [](const vec& v) { return v.expm1(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(ErrorFunctions, Erf) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(erf),
            RESOLVE_OVERLOAD(std::erf),
            [](const vec& v) { return v.erf(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(ErrorFunctions, Erfc) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(erfc),
            RESOLVE_OVERLOAD(std::erfc),
            [](const vec& v) { return v.erfc(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(ErrorFunctions, Erfinv) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(erfinv),
            RESOLVE_OVERLOAD(calc_erfinv),
            [](const vec& v) { return v.erfinv(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }



    TYPED_TEST(LGamma, LGamma) {
        using vec = TypeParam;
        using UVT = UvalueType<vec>;
        UVT tolerance = getDefaultTolerance<UVT>();
        // double: 2e+305  float: 4e+36 (https://sleef.org/purec.xhtml#eg)
        UVT maxCorrect = std::is_same<UVT, float>::value ? (UVT)4e+36 : (UVT)2e+305;
        TestingCase<vec> testCase = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)0}}, true, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)0, (UVT)1000 }}, true, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)1000, maxCorrect }}, true, tolerance})
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(lgamma),
            RESOLVE_OVERLOAD(std::lgamma),
            [](vec v) { return v.lgamma(); },
            testCase);
    }
    TYPED_TEST(InverseTrigonometricReal, ATan2) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(atan2),
            RESOLVE_OVERLOAD(std::atan2),
            [](vec v0, vec v1) {
                return v0.atan2(v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Pow, Pow) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(pow),
            RESOLVE_OVERLOAD(std::pow),
            [](vec v0, vec v1) { return v0.pow(v1); },
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(RealTests, Hypot) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(hypot),
            RESOLVE_OVERLOAD(std::hypot),
            [](vec v0, vec v1) { return v0.hypot(v1); },
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(RealTests, NextAfter) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(nextafter),
            RESOLVE_OVERLOAD(std::nextafter),
            [](vec v0, vec v1) { return v0.nextafter(v1); },
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(Interleave, Interleave) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr auto N = vec::size() * 2LL;
        CACHE_ALIGN VT vals[N];
        CACHE_ALIGN VT interleaved[N];
        auto seed = TestSeed();
        ValueGen<VT> generator(seed);
        for (VT& v : vals) {
            v = generator.get();
        }
        copy_interleave(vals, interleaved);
        auto a = vec::loadu(vals);
        auto b = vec::loadu(vals + vec::size());
        auto cc = interleave2(a, b);
        AssertVec256<vec>(NAME_INFO(Interleave FirstHalf), std::get<0>(cc), vec::loadu(interleaved)).check(true);
        AssertVec256<vec>(NAME_INFO(Interleave SecondHalf), std::get<1>(cc), vec::loadu(interleaved + vec::size())).check(true);
    }
    TYPED_TEST(Interleave, DeInterleave) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr auto N = vec::size() * 2LL;
        CACHE_ALIGN VT vals[N];
        CACHE_ALIGN VT interleaved[N];
        auto seed = TestSeed();
        ValueGen<VT> generator(seed);
        for (VT& v : vals) {
            v = generator.get();
        }
        copy_interleave(vals, interleaved);
        // test interleaved with vals this time
        auto a = vec::loadu(interleaved);
        auto b = vec::loadu(interleaved + vec::size());
        auto cc = deinterleave2(a, b);
        AssertVec256<vec>(NAME_INFO(DeInterleave FirstHalf), std::get<0>(cc), vec::loadu(vals)).check(true);
        AssertVec256<vec>(NAME_INFO(DeInterleave SecondHalf), std::get<1>(cc), vec::loadu(vals + vec::size())).check(true);
    }
    TYPED_TEST(Arithmetics, Plus) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(plus),
            std::plus<VT>(),
            [](const vec& v0, const vec& v1) -> vec {
                return v0 + v1;
            },
            createDefaultBinaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_add_overflow));
    }
    TYPED_TEST(Arithmetics, Minus) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(minus),
            std::minus<VT>(),
            [](const vec& v0, const vec& v1) -> vec {
                return v0 - v1;
            },
            createDefaultBinaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_sub_overflow));
    }
    TYPED_TEST(Arithmetics, Multiplication) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(mult),
            RESOLVE_OVERLOAD(local_multiply),
            [](const vec& v0, const vec& v1) { return v0 * v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true),
            RESOLVE_OVERLOAD(filter_mult_overflow));
    }
    TYPED_TEST(Arithmetics, Division) {
        using vec = TypeParam;
        TestSeed seed;
        test_binary<vec>(
            NAME_INFO(division),
            RESOLVE_OVERLOAD(local_division),
            [](const vec& v0, const vec& v1) { return v0 / v1; },
            createDefaultBinaryTestCase<vec>(seed),
            RESOLVE_OVERLOAD(filter_div_ub));
    }
    TYPED_TEST(Bitwise, BitAnd) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(bit_and),
            RESOLVE_OVERLOAD(local_and),
            [](const vec& v0, const vec& v1) { return v0 & v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Bitwise, BitOr) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(bit_or),
            RESOLVE_OVERLOAD(local_or),
            [](const vec& v0, const vec& v1) { return v0 | v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Bitwise, BitXor) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(bit_xor),
            RESOLVE_OVERLOAD(local_xor),
            [](const vec& v0, const vec& v1) { return v0 ^ v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, Equal) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(== ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::equal_to<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 == v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, NotEqual) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(!= ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::not_equal_to<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 != v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, Greater) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(> ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::greater<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 > v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, Less) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(< ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::less<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 < v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, GreaterEqual) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(>= ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::greater_equal<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 >= v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, LessEqual) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(<= ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::less_equal<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 <= v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(MinMax, Minimum) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(minimum),
            minimum<VT>,
            [](const vec& v0, const vec& v1) {
                return minimum(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, Maximum) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(maximum),
            maximum<VT>,
            [](const vec& v0, const vec& v1) {
                return maximum(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, ClampMin) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(clamp min),
            clamp_min<VT>,
            [](const vec& v0, const vec& v1) {
                return clamp_min(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, ClampMax) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(clamp max),
            clamp_max<VT>,
            [](const vec& v0, const vec& v1) {
                return clamp_max(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, Clamp) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_ternary<vec>(
            NAME_INFO(clamp), clamp<VT>,
            [](const vec& v0, const vec& v1, const vec& v2) {
                return clamp(v0, v1, v2);
            },
            createDefaultTernaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_clamp));
    }
    TYPED_TEST(BitwiseFloatsAdditional, ZeroMask) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        CACHE_ALIGN VT test_vals[vec::size()];
        //all sets will be within 0  2^(n-1)
        auto power_sets = 1 << (vec::size());
        for (int expected = 0; expected < power_sets; expected++) {
            // generate test_val based on expected
            for (int i = 0; i < vec::size(); ++i)
            {
                if (expected & (1 << i)) {
                    test_vals[i] = (VT)0;
                }
                else {
                    test_vals[i] = (VT)0.897;
                }
            }
            int actual = vec::loadu(test_vals).zero_mask();
            ASSERT_EQ(expected, actual) << "Failure Details:\n"
                << std::hex << "Expected:\n#\t" << expected
                << "\nActual:\n#\t" << actual;
        }
    }
    TYPED_TEST(BitwiseFloatsAdditional, Convert) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        using IntVT = at::vec256::int_same_size_t<VT>;

        // verify float to int
        CACHE_ALIGN VT input1[vec::size()];
        CACHE_ALIGN IntVT expected_vals1[vec::size()];
        CACHE_ALIGN IntVT actual_vals1[vec::size()];
        for (int64_t i = 0; i < vec::size(); i++) {
            input1[i] = (VT)i * (VT)2.1 + (VT)0.5;
            expected_vals1[i] = static_cast<IntVT>(input1[i]);
        }
        at::vec256::convert(input1, actual_vals1, vec::size());
        auto expected1 = VecType<IntVT>::loadu(expected_vals1);
        auto actual1 = VecType<IntVT>::loadu(actual_vals1);
        if (AssertVec256<VecType<IntVT>>(NAME_INFO(test_convert_to_int), expected1, actual1).check()) {
          return;
        }

        // verify int to float
        CACHE_ALIGN IntVT input2[vec::size()];
        CACHE_ALIGN VT expected_vals2[vec::size()];
        CACHE_ALIGN VT actual_vals2[vec::size()];
        for (int64_t i = 0; i < vec::size(); i++) {
            input2[i] = (IntVT)i * (IntVT)2 + (IntVT)1;
            expected_vals2[i] = (VT)input2[i];
        }
        at::vec256::convert(input2, actual_vals2, vec::size());
        auto expected2 = vec::loadu(expected_vals2);
        auto actual2 = vec::loadu(actual_vals2);
        AssertVec256<vec>(NAME_INFO(test_convert_to_float), expected2, actual2).check();
    }
    TYPED_TEST(BitwiseFloatsAdditional, Fmadd) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;

        auto test_case = TestingCase<vec>::getBuilder()
          .addDomain(CheckWithinDomains<VT>{
              {{(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}},
              true, getDefaultTolerance<VT>()})
          .setTestSeed(TestSeed());

        test_ternary<vec>(
            NAME_INFO(clamp), RESOLVE_OVERLOAD(local_fmadd),
            [](const vec& v0, const vec& v1, const vec& v2) {
                return at::vec256::fmadd(v0, v1, v2);
            },
            test_case,
            RESOLVE_OVERLOAD(filter_fmadd));
    }
    template<typename vec, typename VT, int64_t mask>
    typename std::enable_if_t<(mask < 0 || mask> 255), void>
    test_blend(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()])
    {
    }
    template<typename vec, typename VT, int64_t mask>
    typename std::enable_if_t<(mask >= 0 && mask <= 255), void>
    test_blend(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()]) {
        // generate expected_val
        int64_t m = mask;
        for (int64_t i = 0; i < vec::size(); i++) {
            expected_val[i] = (m & 0x01) ? b[i] : a[i];
            m = m >> 1;
        }
        // test with blend
        auto vec_a = vec::loadu(a);
        auto vec_b = vec::loadu(b);
        auto expected = vec::loadu(expected_val);
        auto actual = vec::template blend<mask>(vec_a, vec_b);
        auto mask_str = std::string("\nblend mask: ") + std::to_string(mask);
        if (AssertVec256<vec>(std::string(NAME_INFO(test_blend)) + mask_str, expected, actual).check()) return;
        test_blend<vec, VT, mask - 1>(expected_val, a, b);
    }
    template<typename vec, typename VT, int64_t idx, int64_t N>
    std::enable_if_t<(!is_complex<VT>::value && idx == N), bool>
    test_blendv(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], VT mask[vec::size()]) {
        // generate expected_val
        for (int64_t i = 0; i < vec::size(); i++) {
            int64_t hex_mask = 0;
            std::memcpy(&hex_mask, &mask[i], sizeof(VT));
            expected_val[i] = (hex_mask & 0x01) ? b[i] : a[i];
        }
        // test with blendv
        auto vec_a = vec::loadu(a);
        auto vec_b = vec::loadu(b);
        auto vec_m = vec::loadu(mask);
        auto expected = vec::loadu(expected_val);
        auto actual = vec::blendv(vec_a, vec_b, vec_m);
        auto mask_str = std::string("\nblendv mask: ");
        for (int64_t i = 0; i < vec::size(); i++) {
            mask_str += std::to_string(mask[i]) + " ";
        }
        if (AssertVec256<vec>(std::string(NAME_INFO(test_blendv)) + mask_str, expected, actual).check()) {
            return false;
        }
        return true;
    }
    template<typename vec, typename VT, int64_t idx, int64_t N>
    std::enable_if_t<(!is_complex<VT>::value && idx != N), bool>
    test_blendv(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], VT mask[vec::size()]) {
        // shuffle mask and do blendv test
        VT m = mask[idx];
        if (!test_blendv<vec, VT, idx+1, N>(expected_val, a, b, mask)) return false;
        if (m != (VT)0) {
          mask[idx] = (VT)0;
        }
        else {
          int64_t hex_mask = 0xFFFFFFFFFFFFFFFF;
          std::memcpy(&mask[idx], &hex_mask, sizeof(VT));
        }
        if (!test_blendv<vec, VT, idx+1, N>(expected_val, a, b, mask)) return false;
        mask[idx] = m;
        return true;
    }
    template<typename T, int N>
    void blend_init(T(&a)[N], T(&b)[N]) {
        a[0] = (T)1.0;
        b[0] = a[0] + (T)N;
        for (int i = 1; i < N; i++) {
            a[i] = a[i - 1] + (T)(1.0);
            b[i] = b[i - 1] + (T)(1.0);
        }
    }
    template<>
    void blend_init<Complex<float>, 4>(Complex<float>(&a)[4], Complex<float>(&b)[4]) {
        auto add = Complex<float>(1., 100.);
        a[0] = Complex<float>(1., 100.);
        b[0] = Complex<float>(5., 1000.);
        for (int i = 1; i < 4; i++) {
            a[i] = a[i - 1] + add;
            b[i] = b[i - 1] + add;
        }
    }
    template<>
    void blend_init<Complex<double>, 2>(Complex<double>(&a)[2], Complex<double>(&b)[2]) {
        auto add = Complex<double>(1.0, 100.0);
        a[0] = Complex<double>(1.0, 100.0);
        b[0] = Complex<double>(3.0, 1000.0);
        a[1] = a[0] + add;
        b[1] = b[0] + add;
    }
    TYPED_TEST(BitwiseFloatsAdditional, Blendv) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        CACHE_ALIGN VT a[vec::size()];
        CACHE_ALIGN VT b[vec::size()];
        CACHE_ALIGN VT mask[vec::size()] = {0};
        CACHE_ALIGN VT expected_val[vec::size()];
        blend_init(a, b);
        test_blendv<vec, VT, 0, vec::size()>(expected_val, a, b, mask);
    }
    TYPED_TEST(BitwiseFloatsAdditional2, Blend) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        CACHE_ALIGN VT a[vec::size()];
        CACHE_ALIGN VT b[vec::size()];
        CACHE_ALIGN VT expected_val[vec::size()];
        blend_init(a, b);
        constexpr int64_t power_sets = 1LL << (vec::size());
        test_blend<vec, VT, power_sets - 1>(expected_val, a, b);
    }
    template<typename vec, typename VT>
    void test_set(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], int64_t count){
        if (count < 0) return;
        //generate expected_val
        for (int64_t i = 0; i < vec::size(); i++) {
            expected_val[i] = (i < count) ? b[i] : a[i];
        }
        // test with set
        auto vec_a = vec::loadu(a);
        auto vec_b = vec::loadu(b);
        auto expected = vec::loadu(expected_val);
        auto actual = vec::set(vec_a, vec_b, count);

        auto count_str = std::string("\ncount: ") + std::to_string(count);
        if (AssertVec256<vec>(std::string(NAME_INFO(test_set)) + count_str, expected, actual).check()) {
          return;
        }
        test_set<vec, VT>(expected_val, a, b, (count == 0 ? -1 : count / 2));
    }
    TYPED_TEST(BitwiseFloatsAdditional2, Set) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        CACHE_ALIGN VT a[vec::size()];
        CACHE_ALIGN VT b[vec::size()];
        CACHE_ALIGN VT expected_val[vec::size()];
        blend_init(a, b);
        test_set<vec, VT>(expected_val, a, b, vec::size());
    }
    template<typename T>
    std::enable_if_t<!is_complex<T>::value, void>
    arange_init(T& base, T& step) {
        base = (T)5.0;
        step = (T)2.0;
    }
    template<typename T>
    std::enable_if_t<is_complex<T>::value, void>
    arange_init(T& base, T& step) {
       base = T(5.0, 5.0);
       step = T(2.0, 3.0);
    }
    TYPED_TEST(RangeFactories, Arange) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        using UVT = UvalueType<TypeParam>;
        CACHE_ALIGN VT expected_val[vec::size()];
        VT base, step;
        arange_init(base, step);
        for (int64_t i = 0; i < vec::size(); i++) {
            expected_val[i] = base + VT((UVT)i) * step;
        }
        auto expected = vec::loadu(expected_val);
        auto actual = vec::arange(base, step);
        AssertVec256<vec>(NAME_INFO(test_arange), expected, actual).check();
    }
    TEST(ComplexTests, TestComplexFloatImagRealConj) {
        float aa[] = { 1.5488e-28,2.5488e-28,3.5488e-28,4.5488e-28,5.5488e-28,6.5488e-28,7.5488e-28,8.5488e-28 };
        float exp[] = { aa[0],0,aa[2],0,aa[4],0,aa[6],0 };
        float exp3[] = { aa[1],0,aa[3],0,aa[5],0,aa[7],0 };
        float exp4[] = { 1.5488e-28, -2.5488e-28,3.5488e-28,-4.5488e-28,5.5488e-28,-6.5488e-28,7.5488e-28,-8.5488e-28 };
        auto a = vcomplex::loadu(aa);
        auto actual1 = a.real();
        auto actual3 = a.imag();
        auto actual4 = a.conj();
        auto expected1 = vcomplex::loadu(exp);
        auto expected3 = vcomplex::loadu(exp3);
        auto expected4 = vcomplex::loadu(exp4);
        AssertVec256<vcomplex>(NAME_INFO(complex real), expected1, actual1).check();
        AssertVec256<vcomplex>(NAME_INFO(complex imag), expected3, actual3).check();
        AssertVec256<vcomplex>(NAME_INFO(complex conj), expected4, actual4).check();
    }
    TYPED_TEST(QuantizationTests, Quantize) {
        using vec = TypeParam;
        using underlying = ValueType<vec>;
        constexpr int trials = 4000;
        constexpr int min_val = std::numeric_limits<underlying>::min();
        constexpr int max_val = std::numeric_limits<underlying>::max();
        constexpr int el_count = vfloat::size();
        CACHE_ALIGN float unit_float_vec[el_count];
        CACHE_ALIGN underlying expected_qint_vals[vec::size()];
        typename vec::float_vec_return_type  float_ret;
        auto seed = TestSeed();
        //zero point
        ValueGen<int> generator_zp(min_val, max_val, seed);
        //scale
        ValueGen<float> generator_sc(1.f, 15.f, seed.add(1));
        //value
        float minv = static_cast<float>(static_cast<double>(min_val) * 2.0);
        float maxv = static_cast<float>(static_cast<double>(max_val) * 2.0);
        ValueGen<float> gen(minv, maxv, seed.add(2));
        for (int i = 0; i < trials; i++) {
            float scale = generator_sc.get();
            float inv_scale = 1.0f / static_cast<float>(scale);
            auto zero_point_val = generator_zp.get();
            int index = 0;
            for (int j = 0; j < vec::float_num_vecs(); j++) {
                //generate vals
                for (auto& v : unit_float_vec) {
                    v = gen.get();
                    expected_qint_vals[index] = quantize_val<underlying>(scale, zero_point_val, v);
                    index++;
                }
                float_ret[j] = vfloat::loadu(unit_float_vec);
            }
            auto expected = vec::loadu(expected_qint_vals);
            auto actual = vec::quantize(float_ret, scale, zero_point_val, inv_scale);
            if (AssertVec256<vec>(NAME_INFO(Quantize), expected, actual).check()) return;
        } //trials;
    }
    TYPED_TEST(QuantizationTests, DeQuantize) {
        using vec = TypeParam;
        using underlying = ValueType<vec>;
        constexpr bool is_large = sizeof(underlying) > 1;
        constexpr int trials = is_large ? 4000 : std::numeric_limits<underlying>::max() / 2;
        constexpr int min_val = is_large ? -2190 : std::numeric_limits<underlying>::min();
        constexpr int max_val = is_large ? 2199 : std::numeric_limits<underlying>::max();
        CACHE_ALIGN float unit_exp_vals[vfloat::size()];
        CACHE_ALIGN underlying qint_vals[vec::size()];
#if  defined(CHECK_DEQUANT_WITH_LOW_PRECISION)
        std::cout << "Dequant will be tested with relative error " << 1.e-3f << std::endl;
#endif
        auto seed = TestSeed();
        ValueGen<int> generator(min_val, max_val, seed.add(1));
        //scale
        ValueGen<float> generator_sc(1.f, 15.f, seed.add(2));
        for (int i = 0; i < trials; i++) {
            float scale = generator_sc.get();
            int32_t zero_point_val = generator.get();
            float scale_zp_premul = -(scale * zero_point_val);
            vfloat vf_scale = vfloat{ scale };
            vfloat vf_zp = vfloat{ static_cast<float>(zero_point_val) };
            vfloat vf_scale_zp = vfloat{ scale_zp_premul };
            //generate vals
            for (auto& x : qint_vals) {
                x = generator.get();
            }
            //get expected
            int index = 0;
            auto qint_vec = vec::loadu(qint_vals);
            auto actual_float_ret = qint_vec.dequantize(vf_scale, vf_zp, vf_scale_zp);
            for (int j = 0; j < vec::float_num_vecs(); j++) {
                for (auto& v : unit_exp_vals) {
                    v = dequantize_val(scale, zero_point_val, qint_vals[index]);
                    index++;
                }
                vfloat expected = vfloat::loadu(unit_exp_vals);
                const auto& actual = actual_float_ret[j];
#if  defined(CHECK_DEQUANT_WITH_LOW_PRECISION)
                if (AssertVec256<vfloat>(NAME_INFO(DeQuantize), seed, expected, actual).check(false, true, 1.e-3f)) return;
#else
                if (AssertVec256<vfloat>(NAME_INFO(DeQuantize), seed, expected, actual).check()) return;
#endif
            }
        } //trials;
    }
    TYPED_TEST(QuantizationTests, ReQuantizeFromInt) {
        using vec = TypeParam;
        using underlying = ValueType<vec>;
        constexpr int trials = 4000;
        constexpr int min_val = -65535;
        constexpr int max_val = 65535;
        constexpr int el_count = vint::size();
        CACHE_ALIGN c10::qint32 unit_int_vec[el_count];
        CACHE_ALIGN underlying expected_qint_vals[vec::size()];
        typename vec::int_vec_return_type  int_ret;
        auto seed = TestSeed();
        //zero point and value
        ValueGen<int32_t> generator(min_val, max_val, seed);
        //scale
        ValueGen<float> generator_sc(1.f, 15.f, seed.add(1));
        for (int i = 0; i < trials; i++) {
            float multiplier = 1.f / (generator_sc.get());
            auto zero_point_val = generator.get();
            int index = 0;
            for (int j = 0; j < vec::float_num_vecs(); j++) {
                //generate vals
                for (auto& v : unit_int_vec) {
                    v = c10::qint32(generator.get());
                    expected_qint_vals[index] = requantize_from_int<underlying>(multiplier, zero_point_val, v.val_);
                    index++;
                }
                int_ret[j] = vqint::loadu(unit_int_vec);
            }
            auto expected = vec::loadu(expected_qint_vals);
            auto actual = vec::requantize_from_int(int_ret, multiplier, zero_point_val);
            if (AssertVec256<vec>(NAME_INFO(ReQuantizeFromInt), seed, expected, actual).check()) {
                return;
            }
        } //trials;
    }
    TYPED_TEST(QuantizationTests, WideningSubtract) {
        using vec = TypeParam;
        using underlying = ValueType<vec>;
        constexpr bool is_large = sizeof(underlying) > 1;
        constexpr int trials = is_large ? 4000 : std::numeric_limits<underlying>::max() / 2;
        constexpr int min_val = std::numeric_limits<underlying>::min();
        constexpr int max_val = std::numeric_limits<underlying>::max();
        CACHE_ALIGN int32_t unit_exp_vals[vfloat::size()];
        CACHE_ALIGN underlying qint_vals[vec::size()];
        CACHE_ALIGN underlying qint_b[vec::size()];
        typename vec::int_vec_return_type  expected_int_ret;
        auto seed = TestSeed();
        ValueGen<underlying> generator(min_val, max_val, seed);
        for (int i = 0; i < trials; i++) {
            //generate vals
            for (int j = 0; j < vec::size(); j++) {
                qint_vals[j] = generator.get();
                qint_b[j] = generator.get();
                if (std::is_same<underlying, int>::value) {
                    //filter overflow cases
                    filter_sub_overflow(qint_vals[j], qint_b[j]);
                }
            }
            int index = 0;
            auto qint_vec = vec::loadu(qint_vals);
            auto qint_vec_b = vec::loadu(qint_b);
            auto actual_int_ret = qint_vec.widening_subtract(qint_vec_b);
            for (int j = 0; j < vec::float_num_vecs(); j++) {
                for (auto& v : unit_exp_vals) {
                    v = widening_subtract(qint_vals[index], qint_b[index]);
                    index++;
                }
                auto expected = vqint::loadu(unit_exp_vals);
                const auto& actual = actual_int_ret[j];
                if (AssertVec256<vqint>(NAME_INFO(WideningSubtract), seed, expected, actual).check()) return;
            }
        } //trials;
    }
    TYPED_TEST(QuantizationTests, Relu) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr VT min_val = std::numeric_limits<VT>::min();
        constexpr VT max_val = std::numeric_limits<VT>::max();
        constexpr VT fake_zp = sizeof(VT) > 1 ? static_cast<VT>(65535) : static_cast<VT>(47);
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<VT>{ { DomainRange<VT>{min_val, max_val}, DomainRange<VT>{(VT)0, (VT)fake_zp}} })
            .setTestSeed(TestSeed());
        test_binary<vec>(
            NAME_INFO(relu),
            RESOLVE_OVERLOAD(relu),
            [](const vec& v0, const vec& v1) {
                return v0.relu(v1);
            },
            test_case);
    }
    TYPED_TEST(QuantizationTests, Relu6) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr VT min_val = std::numeric_limits<VT>::min();
        constexpr VT max_val = std::numeric_limits<VT>::max();
        constexpr VT fake_zp = sizeof(VT) > 1 ? static_cast<VT>(65535) : static_cast<VT>(47);
        constexpr VT temp = sizeof(VT) > 1 ? static_cast<VT>(12345) : static_cast<VT>(32);
        constexpr VT fake_qsix = fake_zp + temp;
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<VT>{
                {
                    DomainRange<VT>{min_val, max_val},
                        DomainRange<VT>{(VT)0, (VT)fake_zp},
                        DomainRange<VT>{(VT)fake_zp, (VT)fake_qsix}
                }})
            .setTestSeed(TestSeed());
        test_ternary<vec>(
            NAME_INFO(relu6),
            RESOLVE_OVERLOAD(relu6),
            [](/*const*/ vec& v0, const vec& v1, const vec& v2) {
                return  v0.relu6(v1, v2);
            },
            test_case);
    }

#else
#error GTEST does not have TYPED_TEST
#endif
}  // namespace
