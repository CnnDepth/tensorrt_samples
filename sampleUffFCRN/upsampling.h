#ifndef __UPSAMPLING_H__
#define __UPSAMPLING_H__

#include "cudaUtility.h"

/**
 * Function for upsampling image or feature map using nearest neighbor interpolation
 * @ingroup util
 */
cudaError_t cudaResizeNearestNeighbor( float* input, size_t nChannels, size_t inputWidth, size_t inputHeight,
                        float* output, cudaStream_t stream );

/**
 * Function for upsampling image or feature map using bilinear interpolation
 * @ingroup util
 */
cudaError_t cudaResizeBilinear( float* input, size_t nChannels, size_t inputWidth, size_t inputHeight,
                        float* output, cudaStream_t stream );

#endif