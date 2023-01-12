# if has deepstream, build external library for it
if(EXISTS /opt/nvidia/deepstream)
cuda_add_library(nvds_infercustomparser SHARED 
                decode.cu
                prelu.cu
                nvdsinfer_customsparsebbox_Retinaface.cpp)

target_link_libraries(nvds_infercustomparser
    nvinfer_plugin
    nvinfer
    nvparsers
    cudart
    cublas
    stdc++fs)
endif()