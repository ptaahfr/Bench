import time
import numpy as np
from numba import jit

@jit(nopython=True)
def compute_mandelbrot(output, maxVal, centerXY, rangeXY):
    xMin = centerXY[0] - rangeXY[0] / 2
    xMax = centerXY[0] + rangeXY[0] / 2
    yMin = centerXY[1] - rangeXY[1] / 2
    yMax = centerXY[1] + rangeXY[1] / 2

    xStep = (xMax - xMin) / (output.shape[1] - 1)
    yStep = (yMax - yMin) / (output.shape[0] - 1)

    for y in range(output.shape[0]):

        yVal = np.float32(y * yStep + yMin)
        xVal = np.float32(xMin)

        for x in range(output.shape[1]):

            zRe = np.float32(0)
            zIm = np.float32(0)
            zRe2 = np.float32(0)
            zIm2 = np.float32(0)
            val = np.float32(0)

            while True:
                zReIm = zRe * zIm
                (zRe, zIm) = (zRe2 - zIm2 + xVal, zReIm + zReIm + yVal)
                zRe2 = zRe * zRe
                zIm2 = zIm * zIm

                if val >= maxVal or zIm2 + zRe2 >= 4:
                    break

                val += 1
                pass

            output[y][x] = val
            xVal += xStep

            pass
        pass

    pass

def print_result(output):
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            print(chr(32 + (output[y][x] & 63)), end='')
            pass
        print('')
        pass

if __name__ == "__main__":
    width = 100
    height = 25
    maxVal = 256
    repeat = 8

    output = np.ndarray((height, width), dtype=np.int32)

    # Warm-up
    compute_mandelbrot(output, maxVal, (-0.5, 0), (3, 2))

    start = time.time()

    for i in range(repeat):
        compute_mandelbrot(output, maxVal, (-0.5, 0), (3, 2))

    end = time.time()
    print("Python: %g ms" % (1000 * (end - start) / repeat))

    print_result(output)
    
    pass