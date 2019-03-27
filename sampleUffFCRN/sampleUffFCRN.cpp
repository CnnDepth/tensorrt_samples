#include <cassert>
#include <chrono>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <time.h>
#include <unordered_map>
#include <vector>

#include "BatchStreamPPM.h"
#include "NvInferPlugin.h"
#include "NvUffParser.h"
#include "common.h"
#include "fp16.h"

using namespace nvinfer1;
using namespace nvuffparser;
using namespace plugin;

static Logger gLogger;
static samples_common::Args args;

//INT8 Calibration, currently set to calibrate over 500 images
static constexpr int CAL_BATCH_SIZE = 50;
static constexpr int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;

#define RETURN_AND_LOG(ret, severity, message)                                 \
    do                                                                         \
    {                                                                          \
        std::string error_message = "sample_uff_fcrn: " + std::string(message); \
        gLogger.log(ILogger::Severity::k##severity, error_message.c_str());    \
        return (ret);                                                          \
    } while (0)


std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/uff_models/",
                                  "data/ssd/"};
    return locateFile(input, dirs);
}


ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser, nvuffparser::IPluginFactory* pluginFactory,
                                      IInt8Calibrator* calibrator, IHostMemory*& trtModelStream)
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Parse the UFF model to populate the network, then set the outputs.
    INetworkDefinition* network = builder->createNetwork();
    parser->setPluginFactory(pluginFactory);

    std::cout << "Begin parsing model..." << std::endl;
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");

    std::cout << "End parsing model..." << std::endl;

    // Build the engine.
    builder->setMaxBatchSize(maxBatchSize);
    // The _GB literal operator is defined in common/common.h
    builder->setMaxWorkspaceSize(1_GB); // We need about 1GB of scratch space for the plugin layer for batch size 5.
    builder->setHalf2Mode(false);
    if (args.runInInt8)
    {
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(calibrator);
    }

    std::cout << "Begin building engine..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");
    std::cout << "End building engine..." << std::endl;

    // We don't need the network any more, and we can destroy the parser.
    network->destroy();
    parser->destroy();

    // Serialize the engine, then close everything down.
    trtModelStream = engine->serialize();

    builder->destroy();
    shutdownProtobufLibrary();
    return engine;
}


class UpsamplingNearestPlugin : public IPluginExt
{
public:
    UpsamplingNearestPlugin(const Weights *weights, size_t nbWeights)
    {
        std::cout << "Init from weights" << std::endl;
        std::cout << "I have " << nbWeights << " weights" << endl;
    }

    UpsamplingNearestPlugin(const void* data, size_t length)
    {
        std::cout << "Init from data and length" << std::endl;
        const char* d = static_cast<const char*>(data), *a = d;
        read(d, mNbInputChannels);
        read(d, mNbOutputChannels);
        // read the rest from d
        assert(d == a + length);
    }

    ~UpsamplingNearestPlugin()
    {

    }

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        std::cout << "Get output dimensions" << std::endl;
        assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
        assert(mNbInputChannels == inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2]);
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // change output dimensions                                                                               !!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return Dims3(mNbOutputChannels, 1, 1);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override 
    { 
        std::cout << "supports format" << std::endl;
        return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; 
    }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
    {
        std::cout << "configure with format" << std::endl;
        assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
        mDataType = type;
    }

    int initialize() override
    {
        std::cout << "Initialize plugin" << std::endl;
        CHECK(cudnnCreate(&mCudnn));// initialize cudnn and cublas
        CHECK(cublasCreate(&mCublas));
        CHECK(cudnnCreateTensorDescriptor(&mSrcDescriptor));// create cudnn tensor descriptors we need for bias addition
        CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));
        // write below code for custom variables
        return 0;
    }

    virtual void terminate() override
    {
        CHECK(cudnnDestroyTensorDescriptor(mSrcDescriptor));
        CHECK(cudnnDestroyTensorDescriptor(mDstDescriptor));
        CHECK(cublasDestroy(mCublas));
        CHECK(cudnnDestroy(mCudnn));
        // write below code for custom variables
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        return 0;
    }

    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        float onef{1.0f}, zerof{0.0f};
        __half oneh = fp16::__float2half(1.0f), zeroh = fp16::__float2half(0.0f);

        cublasSetStream(mCublas, stream);
        cudnnSetStream(mCudnn, stream);
        //write below code for custom variables  
        return 0;
    }

    virtual size_t getSerializationSize() override
    {
        // size of the layer weights
        return 0;
    }

    virtual void serialize(void* buffer) override
    {
        char* d = static_cast<char*>(buffer), *a = d;

        write(d, mNbInputChannels);
        write(d, mNbOutputChannels);
        write(d, mDataType);
        //write here code for custom variables
        assert(d == a + getSerializationSize());
    }

private:
    size_t type2size(DataType type) { return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half); }

    template<typename T> void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }

    void* copyToDevice(const void* data, size_t count)
    {
        void* deviceData;
        CHECK(cudaMalloc(&deviceData, count));
        CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
        return deviceData;
    }

    void convertAndCopyToDevice(void*& deviceWeights, const Weights& weights)
    {
        if (weights.type != mDataType) // Weights are converted in host memory first, if the type does not match
        {
            size_t size = weights.count*(mDataType == DataType::kFLOAT ? sizeof(float) : sizeof(__half));
            void* buffer = malloc(size);
            for (int64_t v = 0; v < weights.count; ++v)
                if (mDataType == DataType::kFLOAT)
                    static_cast<float*>(buffer)[v] = fp16::__half2float(static_cast<const __half*>(weights.values)[v]);
                else
                    static_cast<__half*>(buffer)[v] = fp16::__float2half(static_cast<const float*>(weights.values)[v]);

            deviceWeights = copyToDevice(buffer, size);
            free(buffer);
        }
        else
            deviceWeights = copyToDevice(weights.values, weights.count * type2size(mDataType));
    }

    void convertAndCopyToBuffer(char*& buffer, const Weights& weights)
    {
        if (weights.type != mDataType)
            for (int64_t v = 0; v < weights.count; ++v)
                if (mDataType == DataType::kFLOAT)
                    reinterpret_cast<float*>(buffer)[v] = fp16::__half2float(static_cast<const __half*>(weights.values)[v]);
                else
                    reinterpret_cast<__half*>(buffer)[v] = fp16::__float2half(static_cast<const float*>(weights.values)[v]);
        else
            memcpy(buffer, weights.values, weights.count * type2size(mDataType));
        buffer += weights.count * type2size(mDataType);
    }

    void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size)
    {
        deviceWeights = copyToDevice(hostBuffer, size);
        hostBuffer += size;
    }

    DataType mDataType{DataType::kFLOAT};
    cudnnHandle_t mCudnn;
    cublasHandle_t mCublas;
    cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
    size_t mNbInputChannels, mNbOutputChannels;
};


class PluginFactory : public nvinfer1::IPluginFactory, public nvuffparser::IPluginFactory
{
public:
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights, const nvuffparser::FieldCollection fc) override
    {
        std::cout << "createPlugin (layerName, weights, nbWeights)" << std::endl;
        std::cout << "name is " << layerName << std::endl;
        assert(isPlugin(layerName));
        if (!strcmp(layerName, "_ResizeBilinear"))
        {
            std::cout << "POOOOOOP" << std::endl;
            mPluginResizeBilinear = std::unique_ptr<UpsamplingNearestPlugin>(new UpsamplingNearestPlugin(weights, nbWeights));
            return mPluginResizeBilinear.get();
        }
        else
        {
            assert(0);
            return nullptr;
        }
    }

    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
    {
        std::cout << "createPlugin (layerName, serialData, serialLength)" << std::endl;
        std::cout << "name is " << layerName << std::endl;
        assert(isPlugin(layerName));
        if (!strcmp(layerName, "_ResizeBilinear"))
        {
            std::cout << "POOOOOOP" << std::endl;
            mPluginResizeBilinear = std::unique_ptr<UpsamplingNearestPlugin>(new UpsamplingNearestPlugin(serialData, serialLength));
            return mPluginResizeBilinear.get();
        }
        else
        {
            assert(0);
            return nullptr;
        }
    }

    bool isPlugin(const char* name) override
    {
        std::cout << "check to be plugin: " << name << std::endl;
        std::cout << ((!strcmp(name, "_ResizeBilinear")) || (!strcmp(name, "_NMS"))) << std::endl;
        return !strcmp(name, "_ResizeBilinear") || !strcmp(name, "_NMS");
    }

    // The application has to destroy the plugin when it knows it's safe to do so.
    void destroyPlugin()
    {
        std::cout << "destroy plugin" << std::endl;
        mPluginResizeBilinear.reset();
    }

    std::unique_ptr<UpsamplingNearestPlugin> mPluginResizeBilinear{nullptr};
};


int main(int argc, char* argv[])
{
    samples_common::parseArgs(args, argc, argv);
    const int N = 1;
    auto fileName = locateFile("fcrn_model.uff");

    auto parser = createUffParser();
    parser->registerInput("Input", DimsCHW(3, 480, 640), UffInputOrder::kNCHW);
    parser->registerOutput("MarkOutput");

    BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);
    IHostMemory* trtModelStream{nullptr};
    Int8EntropyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH, "CalibrationTableFCRN");

    PluginFactory pluginFactorySerialize;
    std::cout << "Plugin factory created" << std::endl;
    ICudaEngine* tmpEngine = loadModelAndCreateEngine(fileName.c_str(), N, parser, &pluginFactorySerialize, &calibrator, trtModelStream);
    assert(tmpEngine != nullptr);
    assert(trtModelStream != nullptr);
    tmpEngine->destroy();
    pluginFactorySerialize.destroyPlugin();

    // save the engine
    std::ofstream p("sampleUffFCRNEngine.trt");
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return 1;
    }
    p.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());
    trtModelStream->destroy();

    // Deserialize the engine.
    std::cout << "*** deserializing" << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    PluginFactory pluginFactory;
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), &pluginFactory);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    return EXIT_SUCCESS;
}