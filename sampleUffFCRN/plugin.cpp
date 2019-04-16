#include "plugin.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include "NvInferPlugin.h"
#include "common.h"
#include "upsampling.h"

int NearestNeighborUpsamplingPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    std::cout << "enquque plugin " << this << std::endl;
    // perform nearest neighbor upsampling using cuda
    std::cout << "Input size: " << mNbInputChannels << ' ' << mInputHeight << ' ' << mInputWidth << std::endl;
    CHECK(cublasSetStream(mCublas, stream));
    CHECK(cudnnSetStream(mCudnn, stream));
    if (mDataType == DataType::kFLOAT)
        CHECK(cudaResizeNearestNeighbor<float>((float*)inputs[0], mNbInputChannels, mInputHeight, mInputWidth, (float*)outputs[0], stream));
    else
        CHECK(cudaResizeNearestNeighbor<__half>((__half*)inputs[0], mNbInputChannels, mInputHeight, mInputWidth, (__half*)outputs[0], stream));
    return 0;
}

IPluginV2* NearestNeighborUpsamplingPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        std::cout << "create plugin using the creator" << std::endl;
        std::cout << "name is " << name << std::endl;
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "nbInputChannels"))
            {
                //assert(fields[i].type == PluginFieldType::kINT32);
                mNbInputChannels = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "inputHeight"))
            {
                //assert(fields[i].type == PluginFieldType::kINT32);
                mInputHeight = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "inputWidth"))
            {
                //assert(fields[i].type == PluginFieldType::kINT32);
                mInputWidth = *(static_cast<const int*>(fields[i].data));
            }
        }
        return new NearestNeighborUpsamplingPlugin(mNbInputChannels, mInputHeight, mInputWidth);
    }