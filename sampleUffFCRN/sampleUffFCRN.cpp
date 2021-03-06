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

#include "NvInferPlugin.h"
#include "NvUffParser.h"
#include "common.h"
#include "upsampling.h"
#include "argsParser.h"
#include "fp16.h"
#include "plugin.h"

using namespace nvinfer1;
using namespace nvuffparser;
using namespace plugin;

static Logger gLogger;
static samplesCommon::Args args;

//INT8 Calibration, currently set to calibrate over 500 images
static constexpr int CAL_BATCH_SIZE = 1;
static constexpr int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;

int INPUT_C = 3;
int INPUT_H = 480;
int INPUT_W = 640;
static constexpr int PPM_INPUT_C = 3;
static constexpr int PPM_INPUT_H = 480;
static constexpr int PPM_INPUT_W = 640;

int OUTPUT_C = 1;
int OUTPUT_H = 480;
int OUTPUT_W = 640;

const char* INPUT_BLOB_NAME = "";
const char* OUTPUT_BLOB_NAME = "";
const char* MODEL_NAME = "";
const char* UFF_FILENAME = "";

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


// Logger for TensorRT info/warning/errors
class iLogger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        //if (severity != Severity::kINFO)
        std::cout << msg << std::endl;
    }
} gLoggerForBuild;


ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser, IHostMemory*& trtModelStream, bool fp16)
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLoggerForBuild);

    // Parse the UFF model to populate the network, then set the outputs.
    INetworkDefinition* network = builder->createNetwork();

    DataType dtype;
    std::cout << "Begin parsing model..." << std::endl;
    if (fp16)
        dtype = nvinfer1::DataType::kHALF;
    else
        dtype = nvinfer1::DataType::kFLOAT;
    if (!parser->parse(uffFile, *network, dtype))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");

    std::cout << "End parsing model..." << std::endl;

    // Build the engine.
    builder->setMaxBatchSize(maxBatchSize);
    // The _GB literal operator is defined in common/common.h
    builder->setMaxWorkspaceSize(1_GB); // We need about 1GB of scratch space for the plugin layer for batch size 5.
    builder->setFp16Mode(fp16);

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

template <typename T>
void doInference(IExecutionContext& context, T* inputData, T* output, size_t batchSize)
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
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(T), cudaMemcpyHostToDevice, stream));

    context.execute(batchSize, &buffers[0]);
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++)
        context.execute(batchSize, &buffers[0]);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    //context.execute(batchSize, &buffers[0]);

    std::cout << "Average inference time over 10 runs: " << total / 10.0 << " ms." << std::endl;

    CHECK(cudaMemcpyAsync(output,
                          buffers[outputIndex],
                          batchSize * OUTPUT_C * OUTPUT_H * OUTPUT_W * sizeof(T),
                          cudaMemcpyDeviceToHost, 
                          stream)
    );
    std::cout << "copying done" << std::endl;
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    std::cout << "Freeing input done" << std::endl;
    CHECK(cudaFree(buffers[outputIndex]));
    std::cout << "Freeing output done" << std::endl;
}

PluginFieldCollection NearestNeighborUpsamplingPluginCreator::mFC{};
std::vector<PluginField> NearestNeighborUpsamplingPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(NearestNeighborUpsamplingPluginCreator);

void processImage(const char* imageFile, const char* outputImageFile, IExecutionContext* context)
{
    // Read sample image
    samplesCommon::PPM<PPM_INPUT_C, PPM_INPUT_H, PPM_INPUT_W> image;
    readPPMFile(imageFile, image);
    std::cout << "PPM image readed" << std::endl;

    // Copy image pixels to buffer
    std::vector<float> data(INPUT_C * INPUT_H * INPUT_W);
    for (int c = 0; c < INPUT_C; ++c)
    {
        for (unsigned j = 0, volChl = INPUT_H * INPUT_W; j < volChl; ++j)
        {
            float pixel_value = float(image.buffer[j * INPUT_C + c]) / 255.0;
            pixel_value = pixel_value * 275.0 - 123.0;
            data[c * volChl + j] = pixel_value;
        }
    }
    std::cout << "Data created" << std::endl;
    std::cout << "Data size: " << data.size() << std::endl;
    std::cout << "First data pixel: " << data[0] << std::endl;

    // Execute engine
    int OUTPUT_SIZE = OUTPUT_C * OUTPUT_H * OUTPUT_W;
    std::vector<float> output(OUTPUT_SIZE);
    doInference<float>(*context, &data[0], &output[0], 1);
    std::cout << "First output pixel: " << output[0] << std::endl;
    float sum_depth = 0;
    float min_depth = 10;
    float max_depth = 0;
    int zero_pixels = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        if (output[i] < min_depth)
            min_depth = output[i];
        if (output[i] > max_depth)
            max_depth = output[i];
        sum_depth += output[i];
        if (output[i] < 1e-5)
            zero_pixels++;
    }
    std::cout << "Min depth: " << min_depth << "; max depth: " << max_depth << std::endl;
    std::cout << "Mean depth: " << sum_depth / OUTPUT_SIZE << std::endl;
    std::cout << "part of zero pixels: " << (float)zero_pixels / (float)OUTPUT_SIZE << std::endl;

    // Write result on disk
    uint8_t* outputPixels = new uint8_t[OUTPUT_SIZE * 3];
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < OUTPUT_H; h++)
            for (int w = 0; w < OUTPUT_W; w++)
            {
                int posCHW = 0 * OUTPUT_H * OUTPUT_W + h * OUTPUT_W + w;
                int posHWC = h * OUTPUT_W * 3 + w * 3 + c;
                outputPixels[posHWC] = uint8_t(output[posCHW] * 25.5);
            }
    std::ofstream outfile(outputImageFile, std::ofstream::binary);
    outfile << "P6"
            << "\n"
            << OUTPUT_W << " " << OUTPUT_H << "\n"
            << 255 << "\n";
    outfile.write(reinterpret_cast<char*>(outputPixels), OUTPUT_SIZE * 3);
    delete outputPixels;
}

int main(int argc, char* argv[])
{
    samplesCommon::parseArgs(args, argc, argv);
    INPUT_C = 3;
    INPUT_H = args.height;
    INPUT_W = args.width;
    OUTPUT_C = 1;
    OUTPUT_H = args.height;
    OUTPUT_W = args.width;
    std::cout << "input h and w: " << INPUT_H << ' ' << INPUT_W << std::endl;
    std::cout << "output h and w: " << OUTPUT_H << ' ' << OUTPUT_W << std::endl;
    INPUT_BLOB_NAME = args.uffInputBlob.c_str();
    OUTPUT_BLOB_NAME = args.outputBlob.c_str();
    MODEL_NAME = args.engineFile.c_str();
    UFF_FILENAME = args.uffModel.c_str();


    auto parser = createUffParser();
    parser->registerInput(INPUT_BLOB_NAME, DimsCHW(INPUT_C, INPUT_H, INPUT_W), UffInputOrder::kNCHW);
    parser->registerOutput(OUTPUT_BLOB_NAME);

    IHostMemory* trtModelStream{nullptr};

    ICudaEngine* tmpEngine = loadModelAndCreateEngine(UFF_FILENAME, 1, parser, trtModelStream, args.fp16);
    assert(tmpEngine != nullptr);
    assert(trtModelStream != nullptr);
    tmpEngine->destroy();

    // save the engine
    if (strlen(MODEL_NAME) > 0)
    {
        std::ofstream p(MODEL_NAME);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return 1;
        }
        p.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());
    }

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
    std::cout << "fp16: " << args.fp16 << std::endl;
    //Read model from file, deserialize it and create runtime, engine and exec context
    /*std::ifstream model( MODEL_NAME, std::ios::binary );

    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(model), {});
    std::size_t modelSize = buffer.size() * sizeof( unsigned char );

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine( buffer.data(), modelSize, nullptr );
    nvinfer1::IExecutionContext* context = engine->createExecutionContext(); 

    if( !context )
    {
    std::cout << "Failed to create execution context" << std::endl;
    return 0;
    }*/

    processImage("bus_preprocessed.ppm", "bus_depth.ppm", context);
    std::cout << "Image processed" << std::endl;
    return EXIT_SUCCESS;
}
