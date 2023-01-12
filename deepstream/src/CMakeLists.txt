include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugins)

AUX_SOURCE_DIRECTORY(${DEEPSTREAM_APP_COMMON_SOURCES} SRC)
add_executable(${PROJECT_NAME}
                deepstream_app_config_parser.cpp
                deepstream_app_main.cpp
                deepstream_app.cpp
                cosine_similarity.cpp
                common.cpp
                plugins/deepstream_dsfacesearch.cpp
                ${SRC})

target_link_libraries(${PROJECT_NAME}
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

set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(gst-dsfacesearch)
