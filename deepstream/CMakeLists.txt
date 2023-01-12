cmake_minimum_required(VERSION 3.2)

set(PROJECT_NAME deepstream-face-recognition)
project(${PROJECT_NAME} VERSION 1.0.0)

add_definitions(-std=c++14)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fpermissive -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED ")

find_package(PkgConfig REQUIRED)
# https://stackoverflow.com/questions/29191855/what-is-the-proper-way-to-use-pkg-config-from-cmake
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(JSON_GLIB REQUIRED json-glib-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(X11_LIB REQUIRED x11)

set(DEEPSTREAM_PATH /opt/nvidia/deepstream/deepstream)
set(DEEPSTREAM_INCLUDE_DIR ${DEEPSTREAM_PATH}/sources/includes)
set(DEEPSTREAM_APP_COMMON_INCLUDE_DIR ${DEEPSTREAM_PATH}/sources/apps/apps-common/includes)
set(DEEPSTREAM_APP_COMMON_SOURCES ${DEEPSTREAM_PATH}/sources/apps/apps-common/src)
set(DEEPSTREAM_LIBRARY_DIR ${DEEPSTREAM_PATH}/lib)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GSTREAMER_INCLUDE_DIRS})
include_directories(${JSON_GLIB_INCLUDE_DIRS})
include_directories(${DEEPSTREAM_INCLUDE_DIR} ${DEEPSTREAM_APP_COMMON_INCLUDE_DIR})

link_directories(
  ${DEEPSTREAM_LIBRARY_DIR}
  ${DEEPSTREAM_LIBRARY_DIR}/gst-plugins)

add_subdirectory(src)
add_subdirectory(nvdsinfer_customparser)

# AUX_SOURCE_DIRECTORY(${DEEPSTREAM_APP_COMMON_SOURCES} SRC)
# add_executable(${PROJECT_NAME}
#                 ${PROJECT_SOURCE_DIR}/src/deepstream_app_config_parser.cpp
#                 ${PROJECT_SOURCE_DIR}/src/deepstream_app_main.cpp
#                 ${PROJECT_SOURCE_DIR}/src/deepstream_app.cpp
#                 ${PROJECT_SOURCE_DIR}/src/cosine_similarity.cpp
#                 ${SRC})

# target_link_libraries(${PROJECT_NAME}
#                 nvinfer
#                 nvinfer_plugin
#                 cudart
#                 cublas
#                 nvparsers
#                 opencv_core
#                 opencv_imgproc
#                 opencv_imgcodecs 
#                 ${GSTREAMER_LIBRARIES}
#                 ${GSTREAMER_VIDEO_LIBRARIES}
#                 ${JSON_GLIB_LIBRARIES}
#                 ${X11_LIB_LIBRARIES}
#                 nvdsgst_meta 
#                 nvds_meta
#                 nvdsgst_helper
#                 nvds_utils
#                 nvdsgst_smartrecord
#                 nvds_msgbroker
#                 nvbufsurface
#                 nvbufsurftransform
#                 boost_system
#                 m
#                 dl
#                 cuda
#                 cublasLt
#                 gstrtspserver-1.0
#                 boost_filesystem)


# # if has deepstream, build external library for it
# if(EXISTS /opt/nvidia/deepstream)
# cuda_add_library(libnvds_infercustomparser SHARED 
#     ${PROJECT_SOURCE_DIR}/nvdsinfer_customparser/decode.cu
#     ${PROJECT_SOURCE_DIR}/nvdsinfer_customparser/prelu.cu
#     ${PROJECT_SOURCE_DIR}/nvdsinfer_customparser/nvdsinfer_customsparsebbox_Retinaface.cpp)

# target_link_libraries(libnvds_infercustomparser
#     nvinfer_plugin
#     nvinfer
#     nvparsers
#     cudart
#     cublas
#     stdc++fs)
# endif()
