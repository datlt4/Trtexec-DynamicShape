#ifndef TRT_EXEC_H
#define TRT_EXEC_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <assert.h>
#include <map>
#include <chrono>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <map>
#include <numeric>
#include <iomanip>
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "fstream"
#include "Logger.h"

struct ParseOnnxConfig
{
    int minBatchSize;
    int minImageChannel;
    int minImageHeight;
    int minImageWidth;
    int optBatchSize;
    int optImageChannel;
    int optImageHeight;
    int optImageWidth;
    int maxBatchSize;
    int maxImageChannel;
    int maxImageHeight;
    int maxImageWidth;
    int workspace{1ULL << 30};
    std::string inputName;
    std::string onnx_dir;
    std::string engine_dir;
    // bool dynamic;
    friend std::ostream &operator<<(std::ostream &os, const ParseOnnxConfig config)
    {
        os << "  --onnx         : " << config.onnx_dir << std::endl
           << "  --engine       : " << config.engine_dir << std::endl
           //    << "  --dynamic      : " << (config.dynamic ? "True" : "False") << std::endl
           << "  --minShape     : " << config.minBatchSize << "x" << config.minImageChannel << "x" << config.minImageHeight << "x" << config.minImageWidth << std::endl
           << "  --optShape     : " << config.optBatchSize << "x" << config.optImageChannel << "x" << config.optImageHeight << "x" << config.optImageWidth << std::endl
           << "  --maxShape     : " << config.maxBatchSize << "x" << config.maxImageChannel << "x" << config.maxImageHeight << "x" << config.maxImageWidth << std::endl;
        return os;
    }
};

void ShowHelpAndExit(const char *szBadOption);
bool ParseCommandLine(int argc, char *argv[], ParseOnnxConfig &config);

extern TrtLoger::Logger *mLogger;
using Severity = nvinfer1::ILogger::Severity;

struct TRTDestroy
{
    template <class T>
    void operator()(T *obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};
template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

template <typename T>
TRTUniquePtr<T> makeUnique(T *t)
{
    return TRTUniquePtr<T>{t};
}

struct Parser
{
    // TrtUniquePtr<nvcaffeparser1::ICaffeParser> caffeParser;
    // TrtUniquePtr<nvuffparser::IUffParser> uffParser;
    TRTUniquePtr<nvonnxparser::IParser> onnxParser;
    operator bool() const
    {
        // return caffeParser || uffParser || onnxParser;
        return !!(onnxParser);
    }
};

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        static TrtLoger::LogLevel map[] = {
            TrtLoger::FATAL, TrtLoger::ERROR, TrtLoger::WARNING, TrtLoger::INFO, TrtLoger::TRACE};
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
        {
            TrtLoger::LogTransaction(mLogger, map[(int)severity], __FILE__, __LINE__, __FUNCTION__).GetStream() << msg;
        }
    }
    nvinfer1::ILogger &getTRTLogger()
    {
        return *this;
    }
};

class TrtExec
{
protected:
    Parser parser;
    TRTUniquePtr<nvinfer1::INetworkDefinition> prediction_network;
    TRTUniquePtr<nvinfer1::ICudaEngine> prediction_engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> prediction_context{nullptr};

    TrtLogger gLogger = TrtLogger();
    int batch_size = 1;
    std::vector<nvinfer1::Dims> prediction_input_dims;
    std::vector<nvinfer1::Dims> prediction_output_dims;

    std::vector<void *> input_buffers; // buffers for input and output data
    std::vector<void *> output_buffers;

    cudaStream_t stream;

public:
    TrtExec(const ParseOnnxConfig &info) : info{info} { cudaStreamCreate(&stream); }
    ~TrtExec() { cudaStreamDestroy(stream); }
    /*virtual*/ bool parseOnnxModel();
    /*virtual*/ bool saveEngine(const std::string &fileName);
    /*virtual*/ bool loadEngine(const std::string &fileName);

private:
    ParseOnnxConfig info;
    int maxBatchSize;
};

#endif // TRT_EXEC_H