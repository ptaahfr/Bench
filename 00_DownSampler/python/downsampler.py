import numpy as np
import time

def downsampler_ns(input):
    return 0.25 * (input[0::2,0::2] + input[1::2,0::2] + input[0::2,0::2] + input[1::2,1::2])

def downsampler(input):
    tmp = input[0::2,] + input[1::2,:]
    return 0.25 * (tmp[:,0::2] + tmp[:,1::2])

def downsampler_inplace(output, input, tmp = None):
    if tmp is None or not tmp.shape[0] == output.shape[0] or not tmp.shape[1] == input.shape[1]: 
        tmp = np.ndarray(shape = [output.shape[0], input.shape[1]])

    np.copyto(tmp, input[0::2,:])
    tmp += input[1::2,:]

    np.copyto(output, tmp[:,0::2])
    output += tmp[:,1::2]

    output *= 0.25
    return tmp

def downsampler_ns_inplace(output, input, tmp = None):
    np.add(input[1::2,0::2], input[0::2,0::2], out=output)
    output += input[0::2,1::2]
    output += input[1::2,1::2]
    output *= 0.25
    pass

def downsampler_ns_inplace_lines(output, input, tmp = None):
    for y in range(output.shape[0]):
        np.add(input[2*y,0::2], input[2*y,1::2], out=output[y,:])
        output[y,:] += input[2*y+1,0::2]
        output[y,:] += input[2*y+1,1::2]
        output[y,:] *= 0.25
        pass

def downsampler_ns_inplace_int16(output, input, tmp = None):
    np.add(input[1::2,0::2], input[0::2,0::2], out=output)
    output += input[0::2,1::2]
    output += input[1::2,1::2]
    output += 2
    np.right_shift(output, 2, out=output)
    pass

if __name__ == "__main__":
    input = np.ones(shape = [1024, 1024])
    count = 100
    
    start = time.time()

    for i in range(count):
        output = downsampler(input)

    end = time.time()

    print("Python: %g ms" % (1000 * (end - start) / count))

    start = time.time()

    for i in range(count):
        output = downsampler_ns(input)

    end = time.time()

    print("Python (ns): %g ms" % (1000 * (end - start) / count))

    output = np.ndarray(shape = [512, 512])
    tmp = None
    start = time.time()

    for i in range(count):
        tmp = downsampler_inplace(output, input, tmp)

    end = time.time()

    print("Python (inplace): %g ms" % (1000 * (end - start) / count))

    output = np.ndarray(shape = [512, 512])
    start = time.time()

    for i in range(count):
        downsampler_ns_inplace(output, input)

    end = time.time()

    print("Python (ns inplace): %g ms" % (1000 * (end - start) / count))

    output = np.ndarray(shape = [512, 512])
    start = time.time()

    for i in range(count):
        downsampler_ns_inplace_lines(output, input)

    end = time.time()

    print("Python (ns inplace lines): %g ms" % (1000 * (end - start) / count))

    input = -np.ones(shape = [1024, 1024], dtype=np.int16)
    output = np.ndarray(shape = [512, 512], dtype=np.int16)
    start = time.time()

    for i in range(count):
        downsampler_ns_inplace_int16(output, input)

    end = time.time()

    print("Python (int16 ns inplace): %g ms" % (1000 * (end - start) / count))

    pass