#ifndef _FACE_DETECTOR_
#define _FACE_DETECTOR_

#include <stdio.h>

#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "logging.h"
#include "decode.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

class RetinaFace {
    public:
        RetinaFace(Logger gLogger, const std::string engineFile, int maxBatchSize, 
                    float nms_threshold, float bbox_threshold);
        ~RetinaFace();
        
        std::vector<decodeplugin::Detection> findFace(cv::Mat &img);
    private:
        void loadEngine(Logger gLogger, const std::string engineFile);

        void preprocess(cv::Mat& img, float* data);
        void inference(float* data, float* prob);
        void postprocess(cv::Mat& img, float* prob);

    private:
        nvinfer1::ICudaEngine* m_engine_;
        nvinfer1::IExecutionContext* m_context_;
        int m_INPUT_H, m_INPUT_W, m_INPUT_SIZE, m_OUTPUT_SIZE, m_maxBatchSize;
        float *m_input, *m_output;
        float m_nms_threshold, m_bbox_threshold;
        std::vector<decodeplugin::Detection> m_outputBbox;

        void* io_buffers_[2];
        cudaStream_t stream;
        int inputIndex, outputIndex;
        const std::string input_layer_name = "data";
        const std::string output_layer_name = "prob";
};

#endif //_FACE_DETECTOR_