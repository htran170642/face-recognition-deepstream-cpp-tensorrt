include_directories(${APP_SOURCES})

# build test for api /recognize
add_executable(test_client
                test_client.cpp
                ${APP_SOURCES}/retinaface.cpp 
                ${APP_SOURCES}/common.cpp
                ${APP_SOURCES}/cosine_similarity.cpp
                ${APP_SOURCES}/arcface.cpp
                ${APP_SOURCES}/webclient.cpp)
                
target_link_libraries(test_client
                        nvinfer 
                        cudart 
                        decodeplugin 
                        preluplugin 
                        cublasLt
                        ${Boost_LIBRARIES}
                        ${OpenCV_LIBRARIES} 
                        ${SQLite3_LIBRARIES}
)

# build test for extract face
add_executable(test_retinaface
                test_retinaface.cpp
                ${APP_SOURCES}/retinaface.cpp 
                ${APP_SOURCES}/common.cpp
                ${APP_SOURCES}/cosine_similarity.cpp
                ${APP_SOURCES}/arcface.cpp
                ${APP_SOURCES}/webclient.cpp)
                
target_link_libraries(test_retinaface
                        nvinfer 
                        cudart 
                        decodeplugin 
                        preluplugin 
                        cublasLt
                        ${Boost_LIBRARIES}
                        ${OpenCV_LIBRARIES} 
                        ${SQLite3_LIBRARIES}
)

# build test for embedding file
add_executable(test_embedding
                test_embedding.cpp
                ${APP_SOURCES}/retinaface.cpp 
                ${APP_SOURCES}/common.cpp
                ${APP_SOURCES}/cosine_similarity.cpp
                ${APP_SOURCES}/arcface.cpp
                ${APP_SOURCES}/webclient.cpp)
                
target_link_libraries(test_embedding
                        nvinfer 
                        cudart 
                        decodeplugin 
                        preluplugin 
                        cublasLt
                        ${Boost_LIBRARIES}
                        ${OpenCV_LIBRARIES} 
                        ${SQLite3_LIBRARIES}
)

# build test for curl
add_executable(test_curl test_curl.cpp)
                
target_link_libraries(test_curl
                        curl
                        cublasLt
                        ${Boost_LIBRARIES}
                        ${OpenCV_LIBRARIES} 
                        ${SQLite3_LIBRARIES}
)
              