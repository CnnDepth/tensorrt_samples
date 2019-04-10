#include "upsampling.h"
#include <iostream>
#include <cassert>

// gpu operation for nearest neighbor upsampling
template <typename T>
__global__ void gpuResizeNearestNeighbor( T* input, int nChannels, int iHeight, int iWidth, T* output)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    const size_t oWidth = 2 * iWidth;
    const size_t oHeight = 2 * iHeight;

    if( x >= nChannels || y >= oHeight || z >= oWidth )
        return;

    const int dy = ((float)y * 0.5);
    const int dz = ((float)z * 0.5);

    const T px = input[x * iWidth * iHeight + dy * iWidth + dz];

    output[x * oWidth * oHeight + y * oWidth + z] = px;
}


// nearest neighbor upsampling
template <typename T>
cudaError_t cudaResizeNearestNeighbor( T* input, size_t nChannels, size_t inputWidth, size_t inputHeight,
                        T* output, cudaStream_t stream )
{
    std::cout << "cudaResizeNearestNeighbor" << std::endl;
    if( !input || !output )
    {
        std::cout << "No input or no output" << std::endl;
        return cudaErrorInvalidDevicePointer;
    }

    if( inputWidth == 0 || inputHeight == 0 )
    {
        std::cout << "Width or height is 0" << std::endl;
        return cudaErrorInvalidValue;
    }

    // launch kernel
    const dim3 blockDim(4, 8, 8);
    const size_t outputWidth = 2 * inputWidth;
    const size_t outputHeight = 2 * inputHeight;
    const dim3 gridDim(iDivUp(nChannels, blockDim.x), iDivUp(outputHeight, blockDim.y), iDivUp(outputWidth, blockDim.z));

    gpuResizeNearestNeighbor<T><<<gridDim, blockDim, 0, stream>>>(input, nChannels, inputHeight, inputWidth, output);

    return CUDA(cudaGetLastError());
}

//gpu operation for bilinear upsampling
template <typename T>
// TODO
__global__ void gpuResizeBilinear( float2 scale, T* input, int iWidth, T* output, int oWidth, int oHeight ) { }

// bilinear upsampling
cudaError_t cudaResizeBilinear( float* input, size_t inputWidth, size_t inputHeight,
                        float* output, size_t outputWidth, size_t outputHeight )
{
    // TODO
    return CUDA(cudaGetLastError());
}

template cudaError_t cudaResizeNearestNeighbor<float>(float*, size_t, size_t, size_t, float*, cudaStream_t);
template cudaError_t cudaResizeNearestNeighbor<__half>(__half*, size_t, size_t, size_t, __half*, cudaStream_t);