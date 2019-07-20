#include <cstdint>
#include <limits>
#include <tuple>

void compute_mandelbrot(int32_t * result, int32_t maxResult, intptr_t stride, int width, int height, float xCenter, float yCenter, float xRange, float yRange)
{
    auto const xMin = xCenter - xRange / 2;
    auto const xMax = xCenter + xRange / 2;
    auto const yMin = yCenter - yRange / 2;
    auto const yMax = yCenter + yRange / 2;

    auto const xStep = (xMax - xMin) / (width - 1);
    auto const yStep = (yMax - yMin) / (height - 1);

    for (int y = 0; y < height; ++y)
    {
        auto const resultLine = &result[y * stride];
        auto const yVal = y * yStep + yMin;

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
}

#include <iostream>

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

int main()
{
    enum { width = 80 };
    enum { height = 25 };
    enum { repeatCount = 1 };
    enum { maxResult = 256 };

    int32_t result[width*height];

    auto start = std::chrono::high_resolution_clock::now();

    for (int repeat = 0; repeat < repeatCount; ++repeat)
    {
        compute_mandelbrot(result, maxResult, width, width, height, 0.0f, 0.0f, 3.0f, 2.0f);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Timing: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f / repeatCount << " ms" << std::endl;

    print_result(result, width, width, height);

    return 0;
}