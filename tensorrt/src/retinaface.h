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
#include "common.h"

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

        static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h);
        static inline cv::Rect get_rect_adapt_landmark(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10]);
        static void nms(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh = 0.4);
        static float iou(float lbox[4], float rbox[4]);
        static bool cmp(const decodeplugin::Detection& a, const decodeplugin::Detection& b);

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