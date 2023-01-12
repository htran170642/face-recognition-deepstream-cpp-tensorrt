
set(APP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR})

cuda_add_library(preluplugin SHARED plugins/prelu.cu)
cuda_add_library(decodeplugin SHARED plugins/decode.cu)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(plugins)

add_executable(${PROJECT_NAME}
                common.cpp
                cosine_similarity.cpp
                db.cpp
                retinaface.cpp 
                arcface.cpp
                app.cpp)

target_link_libraries(${PROJECT_NAME} 
                        nvinfer 
                        cudart 
                        decodeplugin 
                        preluplugin 
                        cublasLt
                        ${Boost_LIBRARIES}
                        ${OpenCV_LIBRARIES} 
                        ${SQLite3_LIBRARIES})


add_subdirectory(test)

