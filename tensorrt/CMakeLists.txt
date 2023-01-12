cmake_minimum_required(VERSION 2.6)

set(PROJECT_NAME app)
project(${PROJECT_NAME})

add_definitions(-std=c++14)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Boost REQUIRED)

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

add_subdirectory(src)