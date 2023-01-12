cmake_minimum_required(VERSION 2.6)

project(retinaface)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    # include_directories(/usr/local/cuda/include)
    # link_directories(/usr/local/cuda/lib64)

    include_directories($ENV{CUDA_DIR}/include)
    link_directories($ENV{CUDA_DIR}/lib64)

    include_directories($ENV{TENSORRT_DIR}/include)
    link_directories($ENV{TENSORRT_DIR}/lib)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(prelu SHARED ${PROJECT_SOURCE_DIR}/prelu.cu)
cuda_add_library(decodeplugin SHARED ${PROJECT_SOURCE_DIR}/decode.cu)
# target_link_libraries(decodeplugin nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(app ${PROJECT_SOURCE_DIR}/retinaface.cpp ${PROJECT_SOURCE_DIR}/arcface.cpp ${PROJECT_SOURCE_DIR}/app.cpp)
target_link_libraries(app nvinfer)
target_link_libraries(app cudart)
target_link_libraries(app decodeplugin)
target_link_libraries(app prelu)
target_link_libraries(app ${OpenCV_LIBRARIES})


add_definitions(-O2 -pthread)

