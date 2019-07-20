#include <vector>
#include <limits>

template <bool USE_DIV, typename TYPE>
inline TYPE downsampler_kernel(TYPE a, TYPE b, TYPE c, TYPE d)
{
    return TYPE(0.25) * (a + b + c + d);
}

template <bool USE_DIV>
inline int16_t downsampler_kernel(int16_t a, int16_t b, int16_t c, int16_t d)
{
    if (USE_DIV)
        return (a + b + c + d) / 4;
    else
        return (a + b + c + d + 2) >> 2;
}

template <bool USE_DIV = false, typename TYPE>
void downsampler(TYPE * out, size_t outWidth, size_t outHeight, TYPE const * in, size_t inWidth, size_t inHeight)
{
    for (size_t y = 0; y < outHeight; ++y)
    {
        TYPE * outLine = out + y * outWidth;
        TYPE const * inLine0 = in + 2 * y * inWidth;
        TYPE const * inLine1 = inLine0 + inWidth;
        for (size_t x = 0; x < inWidth; x += 2)
        {
            outLine[x >> 1] = downsampler_kernel<USE_DIV>(inLine0[x], inLine0[x + 1], inLine1[x], inLine1[x + 1]);
        }
    }
}

#include <immintrin.h>

template <typename TYPE>
void downsampler_avx2(TYPE * out, size_t outWidth, size_t outHeight, TYPE const * in, size_t inWidth, size_t inHeight)
{
    // For simplicity we implement only the case outWidth multiple of 8

    for (int y = 0; y < (int)outHeight; ++y)
    {
        TYPE * outLine = out + y * outWidth;
        TYPE const * inLine0 = in + 2 * y * inWidth;
        TYPE const * inLine1 = inLine0 + inWidth;

        for (size_t x = 0; x < inWidth; x += 16)
        {
            auto inVec0 = _mm256_loadu_ps(inLine0 + x);
            auto inVec1 = _mm256_loadu_ps(inLine1 + x);

            auto inVec2 = _mm256_loadu_ps(inLine0 + x + 8);
            auto inVec3 = _mm256_loadu_ps(inLine1 + x + 8);

            auto outVec0 = _mm256_hadd_ps(inVec0, inVec2);
            auto outVec1 = _mm256_hadd_ps(inVec1, inVec3);

            auto finalShuffled = _mm256_mul_ps(_mm256_set1_ps(0.25f), _mm256_add_ps(outVec0, outVec1));

            _mm256_storeu_ps(outLine + (x >> 1), 
                _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(finalShuffled), _MM_SHUFFLE(3, 1, 2, 0))));
        }
    }
}

#include <chrono>
#include <iostream>

template <typename TYPE>
void fill_input(TYPE * input, size_t width, size_t height)
{
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            input[x + width * y] = (TYPE)((y >> 1) + (x >> 1));
        }
    }
}

int main()
{
    size_t width = 1024;
    size_t height = 1024;

    std::vector<float> input(width * height, 1.0f);
    std::vector<float> output(width * height / 4);

    fill_input(input.data(), width, height);

#ifdef _DEBUG
    size_t count = 1;
#else
    size_t count = 10000;
#endif

    {
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t t = 0; t < count; ++t)
        {
            downsampler(output.data(), width / 2, height / 2, input.data(), width, height);
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "C++: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f / count << " ms" << std::endl;
    }
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t t = 0; t < count; ++t)
        {
            downsampler_avx2(output.data(), width / 2, height / 2, input.data(), width, height);
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "C++ (avx2 intrin): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f / count << " ms" << std::endl;
    }

    std::vector<int16_t> inputI16(width * height);
    std::vector<int16_t> outputI16(width * height / 4);

    fill_input(inputI16.data(), width, height);
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t t = 0; t < count; ++t)
        {
            downsampler<true>(outputI16.data(), width / 2, height / 2, inputI16.data(), width, height);
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "C++ (int16 div): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f / count << " ms" << std::endl;
    }
    
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t t = 0; t < count; ++t)
        {
            downsampler<false>(outputI16.data(), width / 2, height / 2, inputI16.data(), width, height);
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "C++ (int16 rshift): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f / count << " ms" << std::endl;
    }
    return 0;
}