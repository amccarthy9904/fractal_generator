from numba import cuda
import numpy as np

@cuda.jit
def cudakernel0(array,array2):
    thread_position = cuda.grid(1)
    array2[thread_position] = array[thread_position] + 0.5

def mandelbrotkernel(result,xcoord,ycoord,col,limit,n):
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if (t<n):
        x,y,x0,y0
        x0 = x = xcoord[t]
        y0 = y = ycoord[t]
        for i in range(limit):
            if (x * x + y * y >= 4):
                result[t] = col[i]
                return
            zx = x * x - y * y + x0
            y = 2 * x * y + y0
            x = zx
        result[t] = 0
        
array = np.zeros(1024 * 1024, np.float32)
array2 = np.zeros(1024 * 1024, np.float32)
print('Initial array:', array)

print('Kernel launch: cudakernel1[1024, 1024](array)')
cudakernel0[1024, 1024](array,array2)

print('Updated array:', array2)

# Since it is a huge array, let's check that the result is correct:
print('The result is correct:', np.all(array2 == np.zeros(1024 * 1024, np.float32) + 0.5))
# def ASSERT_DRV(err):
#     if isinstance(err, cuda.CUresult):
#         if err != cuda.CUresult.CUDA_SUCCESS:
#             raise RuntimeError("Cuda Error: {}".format(err))
#     elif isinstance(err, nvrtc.nvrtcResult):
#         if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
#             raise RuntimeError("Nvrtc Error: {}".format(err))
#     else:
#         raise RuntimeError("Unknown error type: {}".format(err))

# mandelbrot_Kernel = """\
#     extern "C" __global__
#     void mandelbrot_Kernel(float a, float *x, float *y, float *out, size_t n)
#     {
#         size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
#         if (tid < n) {
#             out[tid] = a * x[tid] + y[tid];
#             }
#         }
#     }
#     """
# # Create program
# err, prog = nvrtc.nvrtcCreateProgram(str.encode(mandelbrot_Kernel), b"mandelbrot_Kernel.cu", 0, [], [])

# # Compile program
# opts = [b"--fmad=false", b"--gpu-architecture=compute_75"]
# err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)

# # Get PTX from compilation
# err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
# ptx = b" " * ptxSize
# err, = nvrtc.nvrtcGetPTX(prog, ptx)
