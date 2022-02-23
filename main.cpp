#include "Trtexec.h"

TrtLoger::Logger *mLogger = TrtLoger::LoggerFactory::CreateConsoleLogger(TrtLoger::INFO);

// ./Trtexec \
//     --onnx ../../model-zoo/fast_pose_res50/fast_res50_256x192_dynamic.onnx \
//     --engine ../../model-zoo/fast_pose_res50/fast_res50_256x192_fp16_dynamic.engine \
//     --inputName "input" \
//     --minShape 1x3x256x192 \
//     --optShape 8x3x256x192 \
//     --maxShape 32x3x256x192 \
//     --workspace 1024



int main(int argc, char **argv)
{
    ParseOnnxConfig config;
    if (ParseCommandLine(argc, argv, config))
    {
        std::unique_ptr<TrtExec> executor = std::make_unique<TrtExec>(config);
        std::cout << config << std::endl;
        // if (config.dynamic)
        {
            executor->parseOnnxModel();
            executor->saveEngine(config.engine_dir);
        }
        MLOG(INFO) << "[ PASSED ]:\n"
                   << config << std::endl;
    }
    else
        MLOG(ERROR) << "[ ERROR ] STOP!!!" << std::endl;
}
