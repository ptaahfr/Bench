#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <iostream>

inline void compute_mandelbrot_line(int32_t * resultLine, int32_t maxResult, int width, float yVal, float xMin, float xStep, std::false_type)
{
    for (int x = 0; x < width; ++x)
    {
        auto const xVal = x * xStep + xMin;
        float reZ = 0;
        float imZ = 0;
        int32_t val = -1;

        do
        {
            std::tie(reZ, imZ) = std::make_tuple(reZ * reZ - imZ * imZ + xVal, 2 * reZ * imZ + yVal);
            val++;
        } while (reZ * reZ + imZ * imZ < 4.0f && val < maxResult);

        resultLine[x] = val;
    }
}

#include <immintrin.h>

inline void store_result(int32_t * resultPacket, __m256i const & result, int32_t count)
{
    // Mask store exist only for float *
     _mm256_maskstore_ps((float *)resultPacket, _mm256_cmpgt_epi32(_mm256_set1_epi32(count), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)), _mm256_castsi256_ps(result));
}

inline void store_result(int32_t * resultPacket, __m256i const & result, std::integral_constant<int, 8>)
{
    _mm256_storeu_si256((__m256i *)resultPacket, result);
}

inline __m256 init_mask(int32_t count)
{
    return _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(count), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
}

inline __m256 init_mask(std::integral_constant<int, 8>)
{
    return _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()));
}

template <typename COUNT>
inline void compute_mandelbrot_packet(int32_t * resultPacket, __m256i const & maxResult, __m256 & xVal, __m256 const & yVal, __m256 const & xStep8, COUNT count)
{
    // Z = ((0,0) ... (0,0))
    auto reZ = _mm256_setzero_ps();
    auto imZ = _mm256_setzero_ps();
    auto reZ2 = _mm256_setzero_ps();
    auto imZ2 = _mm256_setzero_ps();

    // val = [int32x8](-1 .. -1)
    auto val = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());

    // isContinuing = (~0 .. ~0)
    auto isContinuing = init_mask(count);

    do
    {
        auto const reZimZ = _mm256_mul_ps(imZ, reZ);

        // Z^2 = Z^2 + C;
        std::tie(reZ, imZ) = std::make_tuple(
            _mm256_add_ps(_mm256_sub_ps(reZ2, imZ2), xVal),
            _mm256_add_ps(_mm256_add_ps(reZimZ, reZimZ), yVal));
        reZ2 = _mm256_mul_ps(reZ, reZ);
        imZ2 = _mm256_mul_ps(imZ, imZ);

        // val -= (isContinuing ? -1 : 0) <=> val += (isContinuing ? 1 : 0)
        val = _mm256_sub_epi32(val, _mm256_castps_si256(isContinuing));

        // isNotStopping = |Z|^2 >= 4 && maxResult >= val
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

inline void compute_mandelbrot_line(int32_t * resultLine, int32_t maxResult0, int width, float yVal0, float xMin, float xStep0, std::true_type)
{
    auto const xStep8 = _mm256_set1_ps(8 * xStep0);
    auto const yVal = _mm256_set1_ps(yVal0);
    auto const maxResult =_mm256_set1_epi32(maxResult0);

    // xVal = (0 .. 7) * xStep + xMin;
    auto xVal = _mm256_add_ps(
        _mm256_set1_ps(xMin),
        _mm256_mul_ps(
            _mm256_set1_ps(xStep0),
            _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f)));

    int x = 0;
    for (; x <= width - 8; x += 8)
    {
        compute_mandelbrot_packet(&resultLine[x], maxResult, xVal, yVal, xStep8, std::integral_constant<int, 8>());
    }

    if (x < width)
    {
        compute_mandelbrot_packet(&resultLine[x], maxResult, xVal, yVal, xStep8, width - x);
    }
}

template <bool SIMD>
void compute_mandelbrot(int32_t * result, int32_t maxResult, intptr_t stride, int width, int height, float xCenter, float yCenter, float xRange, float yRange, std::integral_constant<bool, SIMD> isSIMD)
{
    auto const xMin = xCenter - xRange / 2;
    auto const xMax = xCenter + xRange / 2;
    auto const yMin = yCenter - yRange / 2;
    auto const yMax = yCenter + yRange / 2;

    auto const xStep = (xMax - xMin) / (width - 1);
    auto const yStep = (yMax - yMin) / (height - 1);

#pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
        auto const resultLine = &result[y * stride];
        auto const yVal = y * yStep + yMin;

        compute_mandelbrot_line(&result[y * stride], maxResult, width, yVal, xMin, xStep, isSIMD);
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
#include <cstring>

int main()
{
    enum { width = 100 };
    enum { height = 25 };
    enum { repeatCount = 100 };
    enum { maxResult = 256 };

    int32_t result[width*height];

    for (int version = 0; version <= 1; ++version)
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::memset(result, 0, width*height*sizeof(result[0]));

        for (int repeat = 0; repeat < repeatCount; ++repeat)
        {
            if (version)
                compute_mandelbrot(result, maxResult, width, width, height, -0.5f, 0.0f, 3.0f, 2.0f, std::true_type());
            else
                compute_mandelbrot(result, maxResult, width, width, height, -0.5f, 0.0f, 3.0f, 2.0f, std::false_type());
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Timing " << (version ? "[AVX2]: " : ": ") << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f / repeatCount << " ms" << std::endl;

        print_result(result, width, width, height);
    }

    return 0;
}