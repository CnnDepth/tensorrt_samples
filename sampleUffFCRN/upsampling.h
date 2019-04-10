#ifndef __UPSAMPLING_H__
#define __UPSAMPLING_H__

#include <cublas_v2.h>
#include "cudaUtility.h"

/**
 * Function for upsampling float32 image or feature map using nearest neighbor interpolation
 * @ingroup util
 */
template<typename T>
cudaError_t cudaResizeNearestNeighbor( T* input, size_t nChannels, size_t inputWidth, size_t inputHeight,
                        T* output, cudaStream_t stream );

/**
 * Function for upsampling image or feature map using bilinear interpolation
 * @ingroup util
 */
template<typename T>
cudaError_t cudaResizeBilinear( T* input, size_t nChannels, size_t inputWidth, size_t inputHeight,
                        T* output, cudaStream_t stream );

#endif