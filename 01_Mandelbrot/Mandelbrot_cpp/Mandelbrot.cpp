#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <iostream>

template <int N>
using is_simd_type = std::integral_constant<int, N>;

using no_simd = std::integral_constant<int, 0>;
using avx2_v1 = std::integral_constant<int, 1>;
using avx2_v2 = std::integral_constant<int, 2>;

inline void compute_mandelbrot_or_julia_line(int32_t * resultLine, int32_t maxResult, int width, float yVal, float xMin, float xStep, bool isJulia, float cx, float cy, no_simd)
{
    for (int x = 0; x < width; ++x)
    {
        auto const xVal = x * xStep + xMin;

        float reZ = isJulia ? xVal : 0;
        float imZ = isJulia ? yVal : 0;

        float const reZInc = isJulia ? cx : xVal;
        float const imZInc = isJulia ? cy : yVal;

        int32_t val = 0;

        do
        {
            std::tie(reZ, imZ) = std::make_tuple(reZ * reZ - imZ * imZ + reZInc, 2 * reZ * imZ + imZInc);
            val++;
        } while (reZ * reZ + imZ * imZ < 4.0f && val < maxResult);

        resultLine[x] = val - 1;
    }
}

#include <immintrin.h>

inline void store_result(int32_t * resultPacket, __m256i const & result, int32_t count)
{
    // Mask store exist only for float *
     _mm256_maskstore_ps((float *)resultPacket, _mm256_cmpgt_epi32(_mm256_set1_epi32(count), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)), _mm256_castsi256_ps(result));
}

inline void store_result(int32_t * resultPacket, __m128i const & result, int32_t count)
{
    // Mask store exist only for float *
     _mm_maskstore_ps((float *)resultPacket, _mm_cmpgt_epi32(_mm_set1_epi32(count), _mm_setr_epi32(0, 1, 2, 3)), _mm_castsi128_ps(result));
}

using eight = std::integral_constant<int, 8>;
using four = std::integral_constant<int, 4>;

inline void store_result(int32_t * resultPacket, __m256i const & result, eight count)
{
    _mm256_storeu_si256((__m256i *)resultPacket, result);
}

inline void store_result(int32_t * resultPacket, __m128i const & result, four count)
{
    _mm_storeu_si128((__m128i *)resultPacket, result);
}

inline __m256 init_mask(int32_t count)
{
    return _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(count), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
}

inline __m256 init_mask(eight count)
{
    __m256i tmp = _mm256_undefined_si256(); // Any register
    return _mm256_castsi256_ps(_mm256_cmpeq_epi32(tmp, tmp));
}

inline __m128 init_mask4(int32_t count)
{
    return _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_set1_epi32(count), _mm_setr_epi32(0, 1, 2, 3)));
}

inline __m128 init_mask4(four count)
{
    __m128i tmp = _mm_undefined_si128();
    return _mm_castsi128_ps(_mm_cmpeq_epi32(tmp, tmp));
}

template <typename COUNT>
inline void compute_mandelbrot_or_julia_packet(int32_t * resultPacket, __m256i const & maxResult, __m256 & xVal, __m256 const & yVal, __m256 const & xStep8, bool isJulia, __m256 const & cx, __m256 const & cy, COUNT count)
{
    // Z = ((0,0) ... (0,0))
    auto reZ = isJulia ? xVal : _mm256_setzero_ps();
    auto imZ = isJulia ? yVal : _mm256_setzero_ps();
    auto reZ2 = _mm256_mul_ps(reZ, reZ);
    auto imZ2 = _mm256_mul_ps(imZ, imZ);

    auto const reZInc = isJulia ? cx : xVal;
    auto const imZInc = isJulia ? cy : yVal;

    // val = [int32x8](-1 .. -1)
    auto val = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());

    // isContinuing = (~0 .. ~0)
    auto isContinuing = init_mask(count);

    do
    {
        auto const reZimZ = _mm256_mul_ps(imZ, reZ);

        // Z^2 = Z^2 + C;
        std::tie(reZ, imZ) = std::make_tuple(
            _mm256_add_ps(_mm256_sub_ps(reZ2, imZ2), reZInc),
            _mm256_add_ps(_mm256_add_ps(reZimZ, reZimZ), imZInc));
        reZ2 = _mm256_mul_ps(reZ, reZ);
        imZ2 = _mm256_mul_ps(imZ, imZ);

        // val -= (isContinuing ? -1 : 0) <=> val += (isContinuing ? 1 : 0)
        val = _mm256_sub_epi32(val, _mm256_castps_si256(isContinuing));

        // isNotStopping = |Z|^2 < 4 && maxResult > val
        auto const isNotStopping = _mm256_and_ps(
                _mm256_cmp_ps(_mm256_add_ps(reZ2, imZ2), _mm256_set1_ps(4.0f), _CMP_LT_OQ),
                _mm256_castsi256_ps(_mm256_cmpgt_epi32(maxResult, val)));

        // isContinuing &&= isNotStopping
        isContinuing = _mm256_and_ps(isNotStopping, isContinuing);

    } while  // any(isContinuing != 0)
        (_mm256_movemask_ps(isContinuing));

    store_result(resultPacket, val, count);

    // xVal += 8 * xStep;
    xVal = _mm256_add_ps(xVal, xStep8);
}


//template <typename COUNT>
//inline void compute_mandelbrot_or_julia_packet(int32_t * resultPacket, __m128i const & maxResult, __m256 & xyVal, __m256 const & xStep4, bool isJulia, __m256 const & cxy, COUNT count)
//{
//    // Z = ((0,0) ... (0,0))
//    auto reImZ = isJulia ? xyVal : _mm256_setzero_ps();
//    auto reImZ2 = _mm256_mul_ps(reImZ, reImZ);
//
//    auto const reZIncImZInc = isJulia ? cxy : xyVal;
//
//    // val = [int32x8](-1 .. -1)
//    auto val = _mm_cmpeq_epi32(maxResult, maxResult);
//
//    // isContinuing = (~0 .. ~0)
//    auto isContinuing = init_mask4(count);
//
//    do
//    {
//        auto const reZimZ = _mm256_mul_ps(reimZ, _mm256_shuffle_ps(reimZ, reimZ, _MM_SHUFFLE(2, 3, 0, 1)));
//
//        // addsub: reim_0 re^2 re^2-im^2_1 2reim_1 
//        // re^2-im^2_0 2reim_0 re^2-im^2_1 2reim_1 
//        _mm256_hadd_ps(reImZ2)
//        // Z^2 = Z^2 + C;
//        std::tie(reZ, imZ) = std::make_tuple(
//            _mm256_add_ps(_mm256_sub_ps(reZ2, imZ2), reZInc),
//            _mm256_add_ps(_mm256_add_ps(reZimZ, reZimZ), imZInc));
//        reZ2 = _mm256_mul_ps(reZ, reZ);
//        imZ2 = _mm256_mul_ps(imZ, imZ);
//
//        // val -= (isContinuing ? -1 : 0) <=> val += (isContinuing ? 1 : 0)
//        val = _mm256_sub_epi32(val, _mm256_castps_si256(isContinuing));
//
//        // isNotStopping = |Z|^2 < 4 && maxResult > val
//        auto const isNotStopping = _mm256_and_ps(
//                _mm256_cmp_ps(_mm256_add_ps(reZ2, imZ2), _mm256_set1_ps(4.0f), _CMP_LT_OQ),
//                _mm256_castsi256_ps(_mm256_cmpgt_epi32(maxResult, val)));
//
//        // isContinuing &&= isNotStopping
//        isContinuing = _mm256_and_ps(isNotStopping, isContinuing);
//
//    } while  // any(isContinuing != 0)
//        (_mm256_movemask_ps(isContinuing));
//
//    store_result(resultPacket, val, count);
//
//    // xVal += 8 * xStep;
//    xVal = _mm256_add_ps(xVal, xStep8);
//}
//
inline void compute_mandelbrot_or_julia_line(int32_t * resultLine, int32_t maxResult0, int width, float yVal0, float xMin, float xStep0, bool isJulia, float cx0, float cy0, avx2_v1)
{
    auto const xStep8 = _mm256_set1_ps(8 * xStep0);
    auto const yVal = _mm256_set1_ps(yVal0);
    auto const maxResult =_mm256_set1_epi32(maxResult0);
    auto const cx = _mm256_set1_ps(cx0);
    auto const cy = _mm256_set1_ps(cy0);

    // xVal = (0 .. 7) * xStep + xMin;
    auto xVal = _mm256_add_ps(
        _mm256_set1_ps(xMin),
        _mm256_mul_ps(
            _mm256_set1_ps(xStep0),
            _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f)));

    int x = 0;
    for (; x <= width - 8; x += 8)
    {
        compute_mandelbrot_or_julia_packet(&resultLine[x], maxResult, xVal, yVal, xStep8, isJulia, cx, cy, eight());
    }

    if (x < width)
    {
        compute_mandelbrot_or_julia_packet(&resultLine[x], maxResult, xVal, yVal, xStep8, isJulia, cx, cy, width - x);
    }
}

//inline void compute_mandelbrot_or_julia_line(int32_t * resultLine, int32_t maxResult0, int width, float yVal0, float xMin, float xStep0, bool isJulia, float cx0, float cy0, avx2_v2)
//{
//    auto const xStep4 = _mm256_setr_ps(4 * xStep0, 0, 4 * xStep0, 0, 4 * xStep0, 0, 4 * xStep0);
//    auto const xyVal = _mm256_set1_ps(yVal0);
//
//    auto const maxResult =_mm_set1_epi32(maxResult0);
//
//    auto const cxy = _mm256_setr_ps(cx0, cy0, cx0, cy0, cx0, cy0, cx0, cy0);
//
//    auto xyVal = _mm256_blend_ps(_mm256_set1_ps(yVal0),
//        _mm256_add_ps(
//        _mm256_set1_ps(xMin),
//        _mm256_mul_ps(
//            _mm256_set1_ps(xStep0),
//            _mm256_setr_ps(0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f))), 1 | 4 | 16 | 64);
//
//    int x = 0;
//    for (; x <= width - 4; x += 4)
//    {
//        compute_mandelbrot_or_julia_packet(&resultLine[x], maxResult, xyVal, xStep4, isJulia, cxy, four()));
//    }
//
//    if (x < width)
//    {
//        compute_mandelbrot_or_julia_packet(&resultLine[x], maxResult, xyVal, xStep4, isJulia, cxy, width - x);
//    }
//}

#include <atomic>
#include <string>

class counter
{
    std::atomic_int32_t c_;
    std::string label_;
public:
    counter(std::string const & label) : c_(0), label_(label) { };
    ~counter()
    {
        std::cout << "Counter [" << label_ << "]: " << c_ << std::endl;
    }
    void operator()() { c_++; }
};

#ifdef USE_COUNTERS
#define COUNTER(str) { static counter c(str); c(); }
#else
#define COUNTER(str)
#endif

inline void compute_mandelbrot_or_julia_line(int32_t * resultLine, int32_t maxResult0, int width, float yVal0, float xMin, float xStep0, bool isJulia, float cx0, float cy0, avx2_v2)
{
    __m128i const refillMask0[] =
    {
        _mm_setr_epi32(-8, -8, -8, -8),
        _mm_setr_epi32( 0, -8, -8, -8),
        _mm_setr_epi32(-8,  0, -8, -8),
        _mm_setr_epi32( 0,  1, -8, -8),
        _mm_setr_epi32(-8, -8,  0, -8),
        _mm_setr_epi32( 0, -8,  1, -8),
        _mm_setr_epi32(-8,  0,  1, -8),
        _mm_setr_epi32( 0,  1,  2, -8),
        _mm_setr_epi32(-8, -8, -8,  0),
        _mm_setr_epi32( 0, -8, -8,  1),
        _mm_setr_epi32(-8,  0, -8,  1),
        _mm_setr_epi32( 0,  1, -8,  2),
        _mm_setr_epi32(-8, -8,  0,  1),
        _mm_setr_epi32( 0, -8,  1,  2),
        _mm_setr_epi32(-8,  0,  1,  2),
        _mm_setr_epi32( 0,  1,  2,  3),
    };

    int const refillPopCnt0[] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 2, 3, 4 };

    auto const yVal = _mm256_set1_ps(yVal0);
    auto const maxResult =_mm256_set1_epi32(maxResult0);
    auto const cx = _mm256_set1_ps(cx0);
    auto const cy = _mm256_set1_ps(cy0);

    // xVal = (0 .. 7) * xStep + xMin;
    auto xVal = _mm256_add_ps(
        _mm256_set1_ps(xMin),
        _mm256_mul_ps(
            _mm256_set1_ps(xStep0),
            _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f)));

    auto const zeroTo7 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    auto xIndex = zeroTo7;
    int xIndexNext = 8;
    auto isContinuing = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(width), xIndex));
    int remainBitMask = _mm256_movemask_ps(isContinuing);

    // Z = ((0,0) ... (0,0))
    auto reZ = isJulia ? xVal : _mm256_setzero_ps();
    auto const imZInit = isJulia ? yVal0 : 0;
    auto imZ = _mm256_set1_ps(imZInit);
    auto reZ2 = _mm256_mul_ps(reZ, reZ);
    auto imZ2 = _mm256_mul_ps(imZ, imZ);

    auto reZInc = isJulia ? cx : xVal;
    auto const imZInc = isJulia ? cy : yVal;

    // val = [int32x8](-1 .. -1)
    auto val = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());

    do
    {
        COUNTER("Main loop");

        auto const reZimZ = _mm256_mul_ps(imZ, reZ);

        // Z^2 = Z^2 + C;
        std::tie(reZ, imZ) = std::make_tuple(
            _mm256_add_ps(_mm256_sub_ps(reZ2, imZ2), reZInc),
            _mm256_add_ps(_mm256_add_ps(reZimZ, reZimZ), imZInc));
        reZ2 = _mm256_mul_ps(reZ, reZ);
        imZ2 = _mm256_mul_ps(imZ, imZ);

        // val -= (isContinuing ? -1 : 0) <=> val += (isContinuing ? 1 : 0)
        val = _mm256_sub_epi32(val, _mm256_castps_si256(isContinuing));

        // isNotStopping = |Z|^2 < 4 && maxResult > val
        auto const isNotStopping = _mm256_and_ps(
            _mm256_cmp_ps(_mm256_add_ps(reZ2, imZ2), _mm256_set1_ps(4.0f), _CMP_LT_OQ),
            _mm256_castsi256_ps(_mm256_cmpgt_epi32(maxResult, val)));

        // isContinuing &&= isNotStopping
        isContinuing = _mm256_and_ps(isNotStopping, isContinuing);

        int currentBitMask = _mm256_movemask_ps(isContinuing);

        if (currentBitMask == remainBitMask)
            continue;

        COUNTER("Refill loop");

        int storeBitMask = remainBitMask ^ currentBitMask;
        int refillBitMask = storeBitMask;

        //if (0 == (storeBitMask & 1));
        //else
        //{
        //    auto const firstXIndex = _mm_cvtsi128_si32(_mm256_extractf128_si256(xIndex, 0));
        //    auto const directStoreMask = _mm256_and_si256(_mm256_cmpgt_epi32(_mm256_set1_epi32(width), xIndex), 
        //        _mm256_cmpeq_epi32(xIndex, _mm256_add_epi32(
        //            _mm256_set1_epi32(firstXIndex), zeroTo7)));

        //    _mm256_maskstore_epi32(resultLine + firstXIndex, directStoreMask, val);

        //    storeBitMask &= ~_mm256_movemask_ps(_mm256_castsi256_ps(directStoreMask));
        //}


#if 1
        using __m256i_or_int32 = union { __m256i v; int32_t s[8]; };

#if 1
            auto xIndexSafe = _mm256_blendv_epi8(xIndex, _mm256_set1_epi32(0), _mm256_castps_si256(isContinuing));
            auto valSafe = _mm256_blendv_epi8(val, _mm256_set1_epi32(resultLine[0]), _mm256_castps_si256(isContinuing));

            __m128i const xIndexLo = _mm256_extractf128_si256(xIndexSafe, 0);
            __m128i const valLo = _mm256_extractf128_si256(valSafe, 0);

#define IMPL(i) \
                resultLine[(uint32_t)_mm_extract_epi32(xIndexLo, i)] = _mm_extract_epi32(valLo, i);

            IMPL(0)
            IMPL(1)
            IMPL(2)
            IMPL(3)
#undef IMPL

            __m128i const xIndexHi = _mm256_extractf128_si256(xIndexSafe, 1);
            __m128i const valHi = _mm256_extractf128_si256(valSafe, 1);

#define IMPL(i) \
                resultLine[(uint32_t)_mm_extract_epi32(xIndexHi, i)] = _mm_extract_epi32(valHi, i);

            IMPL(0)
            IMPL(1)
            IMPL(2)
            IMPL(3)
#undef IMPL

#else

        if (storeBitMask)
        {
#if 1
            __m128i const xIndexLo = _mm256_extractf128_si256(xIndex, 0);
            __m128i const valLo = _mm256_extractf128_si256(val, 0);
            __m128i const xIndexHi = _mm256_extractf128_si256(xIndex, 1);
            __m128i const valHi = _mm256_extractf128_si256(val, 1);

#define IMPL(i) \
            if (storeBitMask & (1 << i)) \
                resultLine[(uint32_t)_mm_extract_epi32(xIndexLo, i)] = _mm_extract_epi32(valLo, i);

            IMPL(0)
            IMPL(1)
            IMPL(2)
            IMPL(3)
#undef IMPL

#define IMPL(i) \
            if (storeBitMask & (1 << (4 + i))) \
                resultLine[(uint32_t)_mm_extract_epi32(xIndexHi, i)] = _mm_extract_epi32(valHi, i);

            IMPL(0)
            IMPL(1)
            IMPL(2)
            IMPL(3)
#undef IMPL
#else
            __m256i_or_int32 xIndexBuf;
            xIndexBuf.v = xIndex;
            __m256i_or_int32 valBuf; 
            valBuf.v = val;

            for (int i = 0; i < 8; ++i)
            {
                if (storeBitMask & (1 << i))
                {
                    resultLine[xIndexBuf.s[i]] = valBuf.s[i]; 
                }
            }
#endif
        }
#endif

        int const refillBitMaskLo = refillBitMask & 15;
        int const refillBitMaskHi = (refillBitMask >> 4) & 15;

        __m256i refillIndex =
            _mm256_insertf128_si256(_mm256_castsi128_si256(refillMask0[refillBitMaskLo]),
                                    _mm_add_epi32(_mm_set1_epi32(refillPopCnt0[refillBitMaskLo]), refillMask0[refillBitMaskHi]), 1);

        auto const j = refillPopCnt0[refillBitMaskLo] + refillPopCnt0[refillBitMaskHi];

        auto const refillMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(refillIndex, _mm256_cmpeq_epi32(refillIndex, refillIndex)));

        xIndex = _mm256_blendv_epi8(xIndex, _mm256_add_epi32(_mm256_set1_epi32(xIndexNext), refillIndex), _mm256_castps_si256(refillMask));
        isContinuing = _mm256_blendv_ps(isContinuing, _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(width), xIndex)), refillMask);
        xVal =  _mm256_add_ps(_mm256_set1_ps(xMin), _mm256_mul_ps(_mm256_set1_ps(xStep0), _mm256_cvtepi32_ps(xIndex)));
        val = _mm256_or_si256(val, _mm256_castps_si256(refillMask));

        if (!isJulia)
        {
            reZ = _mm256_andnot_ps(refillMask, reZ);
            reZInc = xVal;
        }
        else
        {
            reZ = _mm256_blendv_ps(reZ, xVal, refillMask);
        }

        imZ = _mm256_blendv_ps(imZ, _mm256_set1_ps(imZInit), refillMask);

        xIndexNext += j;
#else
        if (currentBitMask == 0)
        {
            if (xIndexNext < width)
            {
                // Full refill
                xIndex = _mm256_add_epi32(_mm256_set1_epi32(xIndexNext), zeroTo7);
                auto xVal = _mm256_add_ps(_mm256_set1_ps(xMin), _mm256_mul_ps(_mm256_set1_ps(xStep0), _mm256_cvtepi32_ps(xIndex)));
                xIndexNext += 8;

                isContinuing = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(width), xIndex));

                reZ = isJulia ? xVal : _mm256_setzero_ps();
                if (!isJulia)
                    reZInc = xVal;
                imZ = _mm256_set1_ps(imZInit);
                val = _mm256_cmpeq_epi32(xIndex, xIndex);

                refillBitMask = 0;
            }
            else
            {
                break;
            }
        }

        if (refillBitMask || storeBitMask)
        {
            auto fillPacketAt = [&](auto indexType, __m128i & reZHalf, __m128i & imZHalf, __m128i & reZIncHalf, __m128i & isContinuingHalf, __m128i & valHalf, __m128i & xIndexHalf)
            {
                enum { index = decltype(indexType)::value, lane = (index >> 2), index_in_lane = index - lane * 4, bit_index = (1 << index) };

                if (0 != (storeBitMask & bit_index))
                {
                    resultLine[_mm_extract_epi32(xIndexHalf, index_in_lane)] = _mm_extract_epi32(valHalf, index_in_lane);
                }

                if (0 != (refillBitMask & bit_index) && (xIndexNext < width))
                {
                    union { float f; uint32_t i; } xValNext;
                    xValNext.f = xIndexNext * xStep0 + xMin;

                    reZHalf = _mm_insert_epi32(reZHalf, isJulia ? xValNext.i : 0, index_in_lane);
                    imZHalf = _mm_insert_epi32(imZHalf, imZInit, index_in_lane);
                    isContinuingHalf = _mm_insert_epi32(isContinuingHalf, -1, index_in_lane);
                    if (false == isJulia)
                        reZIncHalf = _mm_insert_epi32(reZIncHalf, xValNext.i, index_in_lane);

                    valHalf = _mm_insert_epi32(valHalf, -1, index_in_lane);
                    xIndexHalf = _mm_insert_epi32(xIndexHalf, xIndexNext, index_in_lane);

                    xIndexNext++;
                }
            };

            auto reZLo = _mm_castps_si128(_mm256_extractf128_ps(reZ, 0));
            auto imZLo = _mm_castps_si128(_mm256_extractf128_ps(imZ, 0));
            auto reZIncLo = _mm_castps_si128(_mm256_extractf128_ps(reZInc, 0));
            auto isContinuingLo = _mm_castps_si128(_mm256_extractf128_ps(isContinuing, 0));
            auto valLo = _mm256_extractf128_si256(val, 0);
            auto xIndexLo = _mm256_extractf128_si256(xIndex, 0);

            fillPacketAt(std::integral_constant<int, 0>(), reZLo, imZLo, reZIncLo, isContinuingLo, valLo, xIndexLo);
            fillPacketAt(std::integral_constant<int, 1>(), reZLo, imZLo, reZIncLo, isContinuingLo, valLo, xIndexLo);
            fillPacketAt(std::integral_constant<int, 2>(), reZLo, imZLo, reZIncLo, isContinuingLo, valLo, xIndexLo);
            fillPacketAt(std::integral_constant<int, 3>(), reZLo, imZLo, reZIncLo, isContinuingLo, valLo, xIndexLo);

            auto reZHi = _mm_castps_si128(_mm256_extractf128_ps(reZ, 1));
            auto imZHi = _mm_castps_si128(_mm256_extractf128_ps(imZ, 1));
            auto reZIncHi = _mm_castps_si128(_mm256_extractf128_ps(reZInc, 1));
            auto isContinuingHi = _mm_castps_si128(_mm256_extractf128_ps(isContinuing, 1));
            auto valHi = _mm256_extractf128_si256(val, 1);
            auto xIndexHi = _mm256_extractf128_si256(xIndex, 1);

            fillPacketAt(std::integral_constant<int, 4>(), reZHi, imZHi, reZIncHi, isContinuingHi, valHi, xIndexHi);
            fillPacketAt(std::integral_constant<int, 5>(), reZHi, imZHi, reZIncHi, isContinuingHi, valHi, xIndexHi);
            fillPacketAt(std::integral_constant<int, 6>(), reZHi, imZHi, reZIncHi, isContinuingHi, valHi, xIndexHi);
            fillPacketAt(std::integral_constant<int, 7>(), reZHi, imZHi, reZIncHi, isContinuingHi, valHi, xIndexHi);

            reZ = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(reZLo)), _mm_castsi128_ps(reZHi), 1);
            imZ = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(imZLo)), _mm_castsi128_ps(imZHi), 1);
            reZInc = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(reZIncLo)), _mm_castsi128_ps(reZIncHi), 1);
            isContinuing = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(isContinuingLo)), _mm_castsi128_ps(isContinuingHi), 1);
            val = _mm256_insertf128_si256(_mm256_castsi128_si256(valLo), valHi, 1);
            xIndex = _mm256_insertf128_si256(_mm256_castsi128_si256(xIndexLo), xIndexHi, 1);
        }
#endif

        reZ2 = _mm256_mul_ps(reZ, reZ);
        imZ2 = _mm256_mul_ps(imZ, imZ);
        remainBitMask = _mm256_movemask_ps(isContinuing);
    } while (remainBitMask > 0);
}

template <int SIMD>
void compute_mandelbrot_or_julia(int32_t * result, int32_t maxResult, intptr_t stride, int width, int height, float xCenter, float yCenter, float xRange, float yRange, bool isJulia, float cx, float cy, is_simd_type<SIMD> simdType)
{
    auto const xMin = xCenter - xRange / 2;
    auto const xMax = xCenter + xRange / 2;
    auto const yMin = yCenter - yRange / 2;
    auto const yMax = yCenter + yRange / 2;

    auto const xStep = (xMax - xMin) / (width - 1);
    auto const yStep = (yMax - yMin) / (height - 1);

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y)
    {
        auto const resultLine = &result[y * stride];
        auto const yVal = y * yStep + yMin;

        compute_mandelbrot_or_julia_line(&result[y * stride], maxResult, width, yVal, xMin, xStep, isJulia, cx, cy, simdType);
    }
}

void print_result(int32_t const * result, intptr_t stride, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        auto resultLine = &result[y * stride];
        for (int x = 0; x < width; ++x)
        {
            std::cout << (char)(' ' + (resultLine[x] & 63));
        }
        std::cout << '\n';
    }
}

#include <chrono>
#include <vector>
#include <algorithm>
#include <fstream>

static int32_t const hot[] = { 0xb0000, 0xd0000, 0x100000, 0x120000, 0x150000, 0x180000, 0x1a0000, 0x1d0000, 0x200000, 0x220000, 0x250000, 0x270000, 0x2a0000, 0x2d0000, 0x2f0000, 0x320000, 0x350000, 0x370000, 0x3a0000, 0x3c0000, 0x3f0000, 0x420000, 0x440000, 0x470000, 0x4a0000, 0x4c0000, 0x4f0000, 0x510000, 0x540000, 0x570000, 0x590000, 0x5c0000, 0x5f0000, 0x610000, 0x640000, 0x660000, 0x690000, 0x6c0000, 0x6e0000, 0x710000, 0x740000, 0x760000, 0x790000, 0x7b0000, 0x7e0000, 0x810000, 0x830000, 0x860000, 0x890000, 0x8b0000, 0x8e0000, 0x900000, 0x930000, 0x960000, 0x980000, 0x9b0000, 0x9e0000, 0xa00000, 0xa30000, 0xa50000, 0xa80000, 0xab0000, 0xad0000, 0xb00000, 0xb30000, 0xb50000, 0xb80000, 0xba0000, 0xbd0000, 0xc00000, 0xc20000, 0xc50000, 0xc80000, 0xca0000, 0xcd0000, 0xcf0000, 0xd20000, 0xd50000, 0xd70000, 0xda0000, 0xdd0000, 0xdf0000, 0xe20000, 0xe40000, 0xe70000, 0xea0000, 0xec0000, 0xef0000, 0xf20000, 0xf40000, 0xf70000, 0xf90000, 0xfc0000, 0xff0000, 0xff0200, 0xff0500, 0xff0800, 0xff0a00, 0xff0d00, 0xff1000, 0xff1200, 0xff1500, 0xff1700, 0xff1a00, 0xff1d00, 0xff1f00, 0xff2200, 0xff2500, 0xff2700, 0xff2a00, 0xff2c00, 0xff2f00, 0xff3200, 0xff3400, 0xff3700, 0xff3a00, 0xff3c00, 0xff3f00, 0xff4100, 0xff4400, 0xff4700, 0xff4900, 0xff4c00, 0xff4f00, 0xff5100, 0xff5400, 0xff5600, 0xff5900, 0xff5c00, 0xff5e00, 0xff6100, 0xff6400, 0xff6600, 0xff6900, 0xff6b00, 0xff6e00, 0xff7100, 0xff7300, 0xff7600, 0xff7900, 0xff7b00, 0xff7e00, 0xff8000, 0xff8300, 0xff8600, 0xff8800, 0xff8b00, 0xff8e00, 0xff9000, 0xff9300, 0xff9500, 0xff9800, 0xff9b00, 0xff9d00, 0xffa000, 0xffa200, 0xffa500, 0xffa800, 0xffaa00, 0xffad00, 0xffb000, 0xffb200, 0xffb500, 0xffb700, 0xffba00, 0xffbd00, 0xffbf00, 0xffc200, 0xffc500, 0xffc700, 0xffca00, 0xffcc00, 0xffcf00, 0xffd200, 0xffd400, 0xffd700, 0xffda00, 0xffdc00, 0xffdf00, 0xffe100, 0xffe400, 0xffe700, 0xffe900, 0xffec00, 0xffef00, 0xfff100, 0xfff400, 0xfff600, 0xfff900, 0xfffc00, 0xfffe00, 0xffff03, 0xffff07, 0xffff0b, 0xffff0f, 0xffff13, 0xffff17, 0xffff1b, 0xffff1f, 0xffff22, 0xffff26, 0xffff2a, 0xffff2e, 0xffff32, 0xffff36, 0xffff3a, 0xffff3e, 0xffff42, 0xffff46, 0xffff4a, 0xffff4e, 0xffff52, 0xffff56, 0xffff5a, 0xffff5e, 0xffff61, 0xffff65, 0xffff69, 0xffff6d, 0xffff71, 0xffff75, 0xffff79, 0xffff7d, 0xffff81, 0xffff85, 0xffff89, 0xffff8d, 0xffff91, 0xffff95, 0xffff99, 0xffff9d, 0xffffa0, 0xffffa4, 0xffffa8, 0xffffac, 0xffffb0, 0xffffb4, 0xffffb8, 0xffffbc, 0xffffc0, 0xffffc4, 0xffffc8, 0xffffcc, 0xffffd0, 0xffffd4, 0xffffd8, 0xffffdc, 0xffffdf, 0xffffe3, 0xffffe7, 0xffffeb, 0xffffef, 0xfffff3, 0xfffff7, 0xfffffb, 0xffffff };

void write_bmp(char const * fileName, int32_t const * values, intptr_t stride, int width, int height, int32_t const * palette, size_t paletteSize)
{
    // Quick and dirty
    std::ofstream ofs(fileName, std::ios::binary);

    if (ofs)
    {
        auto write_int32 = [&](uint32_t val)
        {
            ofs.put(val);
            ofs.put(val >> 8);
            ofs.put(val >> 16);
            ofs.put(val >> 24);
        };
        
        auto write_int24 = [&](uint32_t val)
        {
            ofs.put(val);
            ofs.put(val >> 8);
            ofs.put(val >> 16);
        };

        auto write_int16 = [&](uint32_t val)
        {
            ofs.put(val);
            ofs.put(val >> 8);
        };

        size_t const fileHeaderSize = 14;
        size_t const infoHeaderSize = 40;

        // Begin: bitmap file header (14 bytes)
        ofs.put('B');
        ofs.put('M');
        write_int32(fileHeaderSize + infoHeaderSize + 3 * width * height); // total file size
        write_int32(0); // reserved
        write_int32(fileHeaderSize + infoHeaderSize); // bitmap data starting address

        // Begin: bitmap info header (40 bytes)
        write_int32(infoHeaderSize);
        write_int32(width);
        write_int32(height);
        write_int16(1); // planar count
        write_int16(24); // bpp
        for (int i = 0; i < 6; ++i) write_int32(0);

        // Begin: bitmap data
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                write_int24(palette[std::min((uint32_t)(paletteSize - 1), (uint32_t)values[y * stride + x])]);
            }
        }
    }
}

#include <sstream>

int main()
{
    enum { width = 1024 };
    enum { height = 1024 };
#ifdef _DEBUG
    enum { repeatCount = 1 };
#else
    enum { repeatCount = 64 };
#endif
    enum { maxResult = 256 };

    bool const isJulia = true;
    float const cx = -0.4f;
    float const cy = 0.6f;
    float const xCenter = 0.0f;
    float const yCenter = 0.0f;
    float const xRange = 2.0f;
    float const yRange = 2.0f;

    std::vector<int32_t> result(width*height);

    for (int version = 0; version <= 2; ++version)
    {
        if (version > 0)
        {
            std::fill(result.begin(), result.end(), 0);
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto compute = [&](auto simdType)
        {
            compute_mandelbrot_or_julia(result.data(), maxResult, width, width, height, xCenter, yCenter, xRange, yRange, isJulia, cx, cy, simdType);
        };

        for (int repeat = 0; repeat < repeatCount; ++repeat)
        {
            switch (version)
            {
            case 2:
                compute(avx2_v2());
                break;
            case 1:
                compute(avx2_v1());
                break;
            default:
                compute(no_simd());
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        char const * label = (version == 0 ? "vanilla" : version == 1 ? "avx2_v1" : "avx2_v2");

        std::cout << "Timing [" << label << "]: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f / repeatCount << " ms" << std::endl;

        if (width < 128 && height < 128)
        {
            print_result(result.data(), width, width, height);
        }

        std::ostringstream oss; oss << "mandelbrot_" << label << ".bmp";
        write_bmp(oss.str().c_str(), result.data(), width, width, height, hot, sizeof(hot)/sizeof(hot[0]));
    }

    return 0;
}