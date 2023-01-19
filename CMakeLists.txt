cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project("Trtexec2")

set(CMAKE_CXX_STANDARD 20)
set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS}")
# set(CMAKE_PREFIX_PATH /usr/local/libtorch)
# set(CMAKE_PREFIX_PATH /usr)

# CUDA
find_package(CUDA REQUIRED)
set(LIB ${LIB} ${CUDA_LIBRARIES})
set(INC ${INC} ${CUDA_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
set(LIB ${LIB} ${OpenCV_LIBRARIES})
set(INC ${INC} ${OpenCV_INCLUDE_DIRS})

# Pytorch
set(TENSORRT_ROOT /usr/local/TensorRT)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
    HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
set(LIB ${LIB} ${TENSORRT_LIBRARY})
set(INC ${INC} ${TENSORRT_INCLUDE_DIR})

# TensorRT executable file
set(SRC ${SRC} Trtexec.cpp)

# Main.cpp
set(SRC ${SRC} main.cpp)
set(APP "TrtExec-bin")

# Target
include_directories(${INC})
add_executable(${APP} ${SRC})
target_link_libraries(${APP} ${LIB})
