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
#include "upsampling.h"
#include "argsParser.h"

using namespace nvinfer1;
using namespace nvuffparser;
using namespace plugin;

static Logger gLogger;
static samplesCommon::Args args;

//INT8 Calibration, currently set to calibrate over 500 images
static constexpr int CAL_BATCH_SIZE = 1;
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
                                      IUffParser* parser, IHostMemory*& trtModelStream)
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Parse the UFF model to populate the network, then set the outputs.
    INetworkDefinition* network = builder->createNetwork();

    std::cout << "Begin parsing model..." << std::endl;
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");

    std::cout << "End parsing model..." << std::endl;

    // Build the engine.
    builder->setMaxBatchSize(maxBatchSize);
    // The _GB literal operator is defined in common/common.h
    builder->setMaxWorkspaceSize(1_GB); // We need about 1GB of scratch space for the plugin layer for batch size 5.
    builder->setHalf2Mode(false);

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


ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser, IInt8Calibrator* calibrator, IHostMemory*& trtModelStream)
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Parse the UFF model to populate the network, then set the outputs.
    INetworkDefinition* network = builder->createNetwork();

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

std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = samplesCommon::volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

void doInference(IExecutionContext& context, float* inputData, float* output, size_t batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    int nbBindings = engine.getNbBindings();
    std::cout << "engine has " << nbBindings << " bindings" << std::endl;

    std::vector<void*> buffers(nbBindings);
    std::vector<std::pair<int64_t, DataType>> buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    for (int i = 0; i < nbBindings; ++i)
    {
        auto bufferSizesOutput = buffersSizes[i];
        buffers[i] = samplesCommon::safeCudaMalloc(bufferSizesOutput.first * samplesCommon::getElementSize(bufferSizesOutput.second));
    }

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings().
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
        //outputIndex1 = outputIndex0 + 1; //engine.getBindingIndex(OUTPUT_BLOB_NAME1);
    std::cout << "Input index: " << inputIndex << std::endl;
    std::cout << "Output index: " << outputIndex << std::endl;
    std::cout << "Input buffer size: " << buffersSizes[inputIndex].first << std::endl;
    std::cout << "Output buffer size: " << buffersSizes[outputIndex].first << std::endl;

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

    auto t_start = std::chrono::high_resolution_clock::now();
    context.execute(batchSize, &buffers[0]);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    std::cout << "Time taken for inference is " << total << " ms." << std::endl;

    CHECK(cudaMemcpyAsync(output,
                          buffers[outputIndex],
                          batchSize * OUTPUT_C * OUTPUT_H * OUTPUT_W * sizeof(float),
                          cudaMemcpyDeviceToHost, 
                          stream)
    );
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


class NearestNeighborUpsamplingPlugin : public IPluginV2
{
public:
    NearestNeighborUpsamplingPlugin(int nbInputChannels, int inputHeight, int inputWidth)
    {
        std::cout << "Init " << this << " from dims" << std::endl;  
        mNbInputChannels = nbInputChannels;
        mInputWidth = inputWidth;
        mInputHeight = inputHeight;
        std::cout << "set input width: " << mInputWidth << std::endl;
    }

    NearestNeighborUpsamplingPlugin(const Weights *weights, size_t nbWeights)
    {
        std::cout << "Init " << this << " from weights" << std::endl;
        std::cout << "I have " << nbWeights << " weights" << endl;
        std::cout << "weights ptr is " << weights << std::endl;
    }

    NearestNeighborUpsamplingPlugin(const void* data, size_t length)
    {
        std::cout << "Init " << this << " from data and length" << std::endl;
        const char* d = static_cast<const char*>(data), *a = d;
        read(d, mNbInputChannels);
        read(d, mInputWidth);
        read(d, mInputHeight);
        read(d, mDataType);
        std::cout << "Readed input width " << mInputWidth << std::endl;
        assert(d == a + length);
    }

    ~NearestNeighborUpsamplingPlugin()
    {
        std::cout << "delete plugin " << this << std::endl;
    }

    int getNbOutputs() const override
    {
        std::cout << "get number of outputs of " << this << std::endl;
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        std::cout << "Get output dimensions of " << this << std::endl;
        std::cout << "input dims are: " << inputs[0].d[0] << ' ' << inputs[0].d[1] << ' ' << inputs[0].d[2] << std::endl;
        assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
        mNbInputChannels = inputs[0].d[0];
        mInputHeight = inputs[0].d[1];
        mInputWidth = inputs[0].d[2];
        std::cout << "set input width " << mInputWidth << std::endl;
        return Dims3(inputs[0].d[0], 2 * inputs[0].d[1], 2 * inputs[0].d[2]);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override 
    { 
        std::cout << "supports format? " << this << std::endl;
        return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; 
    }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
    {
        std::cout << "configure " << this << " with format" << std::endl;
        assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
        mDataType = type;
    }

    int initialize() override
    {
        std::cout << "Initialize plugin " << this << std::endl;
        CHECK(cudnnCreate(&mCudnn));// initialize cudnn and cublas
        CHECK(cublasCreate(&mCublas));
        return 0;
    }

    virtual void terminate() override
    {
        std::cout << "terminate plugin " << this << std::endl;
        CHECK(cublasDestroy(mCublas));
        CHECK(cudnnDestroy(mCudnn));
        // write below code for custom variables
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        std::cout << "get workspace size of " << this << std::endl;
        return 0;
    }

    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        std::cout << "enquque plugin " << this << std::endl;
        // perform nearest neighbor upsampling using cuda
        CHECK(cublasSetStream(mCublas, stream));
        CHECK(cudnnSetStream(mCudnn, stream));
        CHECK(cudaResizeNearestNeighbor((float*)inputs[0], mNbInputChannels, mInputWidth, mInputHeight, (float*)outputs[0], stream));
        return 0;
    }

    virtual size_t getSerializationSize() const override
    {
        // 3 size_t values: input width, input height, and number of channels
        // and one more value for data type
        return sizeof(DataType) + 3 * sizeof(int);
    }

    virtual void serialize(void* buffer) const override
    {
        std::cout << "serialize " << this << std::endl;
        char* d = static_cast<char*>(buffer), *a = d;

        std::cout << "this is " << this << std::endl;
        std::cout << "Write input width " << mInputWidth << std::endl;
        write(d, mNbInputChannels);
        write(d, mInputWidth);
        write(d, mInputHeight);
        write(d, mDataType);
        assert(d == a + getSerializationSize());
    }

    const char* getPluginType() const override 
    { 
        std::cout << "get type of " << this << std::endl;
        return "ResizeNearestNeighbor";
    }

    const char* getPluginVersion() const override 
    { 
        std::cout << "get version of " << this << std::endl;
        return "1";
    }

    void destroy() override { delete this; }

    IPluginV2* clone() const override
    {
        return new NearestNeighborUpsamplingPlugin(mNbInputChannels, mInputHeight, mInputWidth);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    size_t type2size(DataType type) { return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half); }

    template<typename T> void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> T read(const char*& buffer, T& val) const
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
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
    int mNbInputChannels=0, mInputWidth=0, mInputHeight=0;
    std::string mNamespace = "";
};


class NearestNeighborUpsamplingPluginCreator: public IPluginCreator
{
public:
    NearestNeighborUpsamplingPluginCreator()
    {
        std::cout << "Create plugin creator" << std::endl;
        mPluginAttributes.emplace_back(PluginField("nbInputChannels", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("inputHeight", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("inputWidth", nullptr, PluginFieldType::kINT32, 1));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    ~NearestNeighborUpsamplingPluginCreator() {}

    const char* getPluginName() const override 
    {
        std::cout << "get plugin name" << std::endl;
        return "ResizeNearestNeighbor"; 
    }

    const char* getPluginVersion() const override
    {
        std::cout << "get plugin version" << std::endl;
        return "1";
    }

    const PluginFieldCollection* getFieldNames() override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
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

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {
        std::cout << "deserialize plugin using the creator" << std::endl;
        //This object will be deleted when the network is destroyed, which will
        //call Concat::destroy()
        return new NearestNeighborUpsamplingPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }
private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace = "";
    int mNbInputChannels, mInputHeight, mInputWidth;
};

PluginFieldCollection NearestNeighborUpsamplingPluginCreator::mFC{};
std::vector<PluginField> NearestNeighborUpsamplingPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(NearestNeighborUpsamplingPluginCreator);

int main(int argc, char* argv[])
{
    samplesCommon::parseArgs(args, argc, argv);
    const int N = 1;
    auto fileName = locateFile("test_upsampling.uff");

    auto parser = createUffParser();
    parser->registerInput("Placeholder", DimsCHW(3, 240, 320), UffInputOrder::kNCHW);
    parser->registerOutput("MarkOutput_0");

    BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);
    IHostMemory* trtModelStream{nullptr};

    ICudaEngine* tmpEngine = loadModelAndCreateEngine(fileName.c_str(), N, parser, trtModelStream);
    assert(tmpEngine != nullptr);
    assert(trtModelStream != nullptr);
    tmpEngine->destroy();

    // save the engine
    /*std::ofstream p("sampleUffFCRNEngine.trt");
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return 1;
    }
    p.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());
    */

    // Deserialize the engine.
    std::cout << "*** deserializing" << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    std::cout << "Runtime created" << std::endl;
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    std::cout << "engine created" << std::endl;
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Read sample image
    samplesCommon::PPM<INPUT_C, INPUT_H, INPUT_W> image;
    readPPMFile("bus.ppm", image);
    std::cout << "PPM image readed" << std::endl;
    vector<float> data(N * INPUT_C * INPUT_H * INPUT_W);
    for (int c = 0; c < INPUT_C; ++c)
    {
        for (unsigned j = 0, volChl = INPUT_H * INPUT_W; j < volChl; ++j)
        {
            data[c * volChl + j] = float(image.buffer[j * INPUT_C + c]) / 255.0;
        }
    }
    std::cout << "Data created" << std::endl;
    std::cout << "Data size: " << data.size() << std::endl;
    std::cout << "First data pixel: " << data[0] << std::endl;

    // Execute engine
    std::vector<float> output(N * OUTPUT_C * OUTPUT_W * OUTPUT_H);
    std::cout << "first output pixel: " << output[0] << std::endl;
    doInference(*context, &data[0], &output[0], 1);
    std::cout << "first output pixel: " << output[0] << std::endl;

    // Write result on disk
    int OUTPUT_SIZE = N * OUTPUT_C * OUTPUT_H * OUTPUT_W;
    uint8_t* outputPixels = new uint8_t[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++)
        outputPixels[i] = uint8_t(output[i] * 255.0);
    std::ofstream outfile("bus_upsampled.ppm", std::ofstream::binary);
    outfile << "P6"
            << "\n"
            << OUTPUT_W << " " << OUTPUT_H << "\n"
            << 255 << "\n";
    outfile.write(reinterpret_cast<char*>(outputPixels), OUTPUT_SIZE);
    delete outputPixels;
    return EXIT_SUCCESS;
}