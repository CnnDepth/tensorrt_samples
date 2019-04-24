#ifndef TENSORRT_ARGS_PARSER_H
#define TENSORRT_ARGS_PARSER_H

#include <vector>
#include <string>
#include <cstdlib>
#ifdef _MSC_VER
#include "..\common\windows\getopt.h"
#else
#include <getopt.h>
#endif
#include <iostream>

namespace samplesCommon
{

//!
//! \brief The SampleParams structure groups the basic parameters required by
//!        all sample networks.
//!
struct SampleParams
{
    int batchSize; //!< Number of inputs in a batch
    int dlaCore{-1};
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
};

//!
//! \brief The CaffeSampleParams structure groups the additional parameters required by
//!         networks that use caffe
//!
struct CaffeSampleParams : public SampleParams
{
    std::string prototxtFileName; //!< Filename of prototxt design file of a network
    std::string weightsFileName;  //!< Filename of trained weights file of a network
};

//!
//! /brief Struct to maintain command-line arguments.
//!
struct Args
{
    bool runInInt8{false};
    bool help{false};
    bool fp16{false};
    std::string uffModel{""};
    std::string uffInputBlob{""};
    std::string outputBlob{""};
    std::string engineFile{""};
    int height{480};
    int width{640};
    int useDLACore{-1};
    std::vector<std::string> dataDirs;
};

//!
//! \brief Populates the Args struct with the provided command-line parameters.
//!
//! \throw invalid_argument if any of the arguments are not valid
//!
//! \return boolean If return value is true, execution can continue, otherwise program should exit
//!
inline bool parseArgs(Args& args, int argc, char* argv[])
{
    while (1)
    {
        std::cout << "while" << std::endl;
        int arg;
        static struct option long_options[] = {
            {"help", no_argument, 0, 'h'},
            {"uff", required_argument, 0, 'U'},
            {"uffInput", required_argument, 0, 'I'},
            {"height", required_argument, 0, 'H'},
            {"width", required_argument, 0, 'W'},
            {"output", required_argument, 0, 'O'},
            {"engine", required_argument, 0, 'e'},
            {"datadir", required_argument, 0, 'd'},
            {"int8", no_argument, 0, 'i'},
            {"useDLACore", required_argument, 0, 'u'},
            {"fp16", no_argument, 0, 'f'},
            {nullptr, 0, nullptr, 0}};
        int option_index = 0;
        arg = getopt_long_only(argc, argv, "hd:iu", long_options, &option_index);
        std::cout << "option index: " << option_index << std::endl;
        std::cout << "arg: " << arg << std::endl;
        if (arg == -1)
            break;

        switch (arg)
        {
        case 'h':
            args.help = true;
            return false;
        case 'd':
            if (optarg)
                args.dataDirs.push_back(optarg);
            else
            {
                std::cerr << "ERROR: --datadir requires option argument" << std::endl;
                return false;
            }
            break;
        case 'i':
            args.runInInt8 = true;
            break;
        case 'u':
            if (optarg)
                args.useDLACore = std::stoi(optarg);
            break;
        case 'f':
            args.fp16 = true;
            break;
        case 'U':
            if (optarg)
                args.uffModel = optarg;
            else
            {
                std::cerr << "ERROR: --uff requires option argument" << std::endl;
                return false;
            }
            break;
        case 'I':
            if (optarg)
                args.uffInputBlob = optarg;
            else
            {
                std::cerr << "ERROR: --uffInput requires option argument" << std::endl;
                return false;
            }
            break;
        case 'O':
            if (optarg)
                args.outputBlob = optarg;
            else
            {
                std::cerr << "ERROR: --output requires option argument" << std::endl;
                return false;
            }
            break;
        case 'H':
            std::cout << "parsing input height" << std::endl;
            std::cout << "optarg is: " << optarg << std::endl;
            if (optarg)
                args.height = atoi(optarg);
            else
            {
                std::cerr << "ERROR: --height requires option argument" << std::endl;
                return false;
            }
            break;
        case 'W':
            std::cout << "parsing input width" << std::endl;
            std::cout << "optarg is: " << optarg << std::endl;
            if (optarg)
                args.width = atoi(optarg);
            else
            {
                std::cerr << "ERROR: --width requires option argument" << std::endl;
                return false;
            }
            break;
        case 'e':
            if (optarg)
                args.engineFile = optarg;
            else
            {
                std::cerr << "ERROR: --engine requires option argument" << std::endl;
                return false;
            }
            break;
        default:
            return false;
        }
    }
    return true;
}

} // namespace samplesCommon

#endif // TENSORRT_ARGS_PARSER_H
