#ifndef _ARCFACE_H_
#define _ARCFACE_H_

#include <stdio.h>

#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "logging.h"
#include "decode.h"
#include "common.h"
#include "cosine_similarity.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#define MAX_FACES_PER_IMAGE 5
#define NUM_EMBEDDING 512
#define REG_INPUT_WIDTH 112
#define REG_INPUT_HEIGHT 112

struct CroppedFace {
    cv::Mat face;
    cv::Mat faceMat;
    int x1, y1, x2, y2;
};

void getCroppedFaces(cv::Mat frame, std::vector<decodeplugin::Detection> &outputBbox, std::vector<struct CroppedFace> &croppedFaces);

class ArcFaceR100 {
    public:
        ArcFaceR100(Logger gLogger, const std::string engineFile, int maxBatchSize);
        ~ArcFaceR100();

        void addEmbedding(const std::string className, float embedding[]);
        void addEmbedding(const std::string className, std::vector<float> embedding);
        void extractFeature(cv::Mat &frame, float *m_output);
        void extractFeature(cv::Mat frame, std::vector<decodeplugin::Detection> &outputBbox);
        float *matchFeature();
        std::tuple<std::vector<std::string>, std::vector<float>> getOutputs(float *output_sims);
        void initKnownEmbeds(int num);
        void initMatMul();

        std::vector<struct CroppedFace> croppedFaces;
        static int classCount;
        
    private:
        void loadEngine(Logger gLogger, const std::string engineFile);

        void preprocessFace(cv::Mat& img);
        void preprocessFaces();

        void inference(float* data, float* prob);
        void inference(float *data, float *prob, int batchSize);
        void normalize(float* output);

    private:
        nvinfer1::ICudaEngine* m_engine_;
        nvinfer1::IExecutionContext* m_context_;
        int m_INPUT_H, m_INPUT_W, m_INPUT_SIZE, m_OUTPUT_SIZE, m_maxBatchSize;
        cv::Mat m_input;
        float *m_output, *m_outputs, *m_embed, *m_embeds, *m_knownEmbeds;
        std::vector<std::string> classNames;

        void* io_buffers_[2];
        cudaStream_t stream;
        int inputIndex, outputIndex;
        const std::string input_layer_name = "data";
        const std::string output_layer_name = "prob";

        CosineSimilarityCalculator matmul;
};

#endif //_ARCFACE_H_