include_directories(${COMMON_DIR})

add_definitions(-shared -Wl,-no-undefined)

cuda_add_library(nvdsgst_dsfacesearch SHARED
                gstdsfacesearch_optimized.cpp
                webclient.cpp
                ${COMMON_DIR}/common.cpp
                )

target_link_libraries(nvdsgst_dsfacesearch
                nvinfer
                nvinfer_plugin
                cudart
                cublas
                nvparsers
                opencv_core
                opencv_imgproc
                opencv_imgcodecs 
                ${GSTREAMER_LIBRARIES}
                ${GSTREAMER_VIDEO_LIBRARIES}
                ${JSON_GLIB_LIBRARIES}
                ${X11_LIB_LIBRARIES}
                nvdsgst_meta 
                nvds_meta
                nvdsgst_helper
                nvds_utils
                nvdsgst_smartrecord
                nvds_msgbroker
                nvbufsurface
                nvbufsurftransform
                boost_system
                m
                dl
                cuda
                cublasLt
                gstrtspserver-1.0
                boost_filesystem)