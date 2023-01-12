#include "retinaface.h"
#include "common.hpp"

RetinaFace::RetinaFace(Logger gLogger, const std::string engineFile, int maxBatchSize, 
                    float nms_threshold, float bbox_threshold) {
  
  m_INPUT_H = decodeplugin::INPUT_H;  // H, W must be able to  be divided by 32.
  m_INPUT_W = decodeplugin::INPUT_W;
  m_INPUT_SIZE = 3 * m_INPUT_H * m_INPUT_W;
  m_OUTPUT_SIZE = (m_INPUT_H / 8 * m_INPUT_W / 8 + m_INPUT_H / 16 * m_INPUT_W / 16 + m_INPUT_H / 32 * m_INPUT_W / 32) * 2  * 15 + 1;
  m_maxBatchSize = maxBatchSize;
  m_nms_threshold = nms_threshold;
  m_bbox_threshold = bbox_threshold;
  m_input = new float[m_maxBatchSize * m_INPUT_SIZE];
  m_output = new float[m_maxBatchSize * m_OUTPUT_SIZE];

  // load engine from .engine file
  loadEngine(gLogger, engineFile);
}

RetinaFace::~RetinaFace() {
  if (m_context_) {
    m_context_->destroy();
  }
  m_context_ = NULL;

  if (m_engine_) {
    m_engine_->destroy();
  }
  m_engine_ = NULL;
}

std::vector<decodeplugin::Detection> RetinaFace::findFace(cv::Mat &img) {
  preprocess(img, m_input);
  inference(m_input, m_output);
  postprocess(img, m_output);
  return m_outputBbox;
}


void RetinaFace::loadEngine(Logger gLogger, const std::string engineFile) {

  if (fileExists(engineFile)) {
    std::cout << "[INFO] Loading RetinaFace Engine...\n";
    std::ifstream file(engineFile, std::ios::binary);
    char* trtModelStream = NULL;
    uint64_t size = 0;
    if (file.good()) {
      file.seekg(0, file.end);
      size = file.tellg();
      file.seekg(0, file.beg);
      trtModelStream = new char[size];
      assert(trtModelStream);
      file.read(trtModelStream, size);
      file.close();
    }

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    m_engine_ = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(m_engine_ != nullptr);
    delete[] trtModelStream;

    m_context_ = m_engine_->createExecutionContext();
    assert(m_context_ != nullptr);

    // Create stream
    checkCudaStatus(cudaStreamCreate(&stream));

    inputIndex = m_engine_->getBindingIndex(input_layer_name.c_str());
    outputIndex = m_engine_->getBindingIndex(output_layer_name.c_str());

    checkCudaStatus(cudaMalloc(&io_buffers_[inputIndex], m_maxBatchSize * m_INPUT_SIZE * sizeof(float)));

    checkCudaStatus(cudaMalloc(&io_buffers_[outputIndex], m_maxBatchSize * m_OUTPUT_SIZE * sizeof(float)));
  } else {
      throw std::logic_error("Cant find engine file");
  }
  
}

void RetinaFace::preprocess(cv::Mat& img, float* data) {
  cv::Mat pr_img = preprocess_img(img, m_INPUT_W, m_INPUT_H);
  cv::imwrite("preprocessed.jpg", pr_img);

  for (int b = 0; b < m_maxBatchSize; b++) {
    float *p_data = &data[b * 3 * m_INPUT_H * m_INPUT_W];
    for (int i = 0; i < m_INPUT_H * m_INPUT_W; i++) {
        p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
        p_data[i + m_INPUT_H * m_INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
        p_data[i + 2 * m_INPUT_H * m_INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
    }
  }
}

void RetinaFace::inference(float* data, float* prob) {
  CHECK(cudaMemcpyAsync(io_buffers_[inputIndex], data, m_maxBatchSize * m_INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
  m_context_->enqueue(m_maxBatchSize, io_buffers_, stream, nullptr);
  CHECK(cudaMemcpyAsync(prob, io_buffers_[outputIndex], m_maxBatchSize * m_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void RetinaFace::postprocess(cv::Mat& img, float* prob) {
  m_outputBbox.clear();
    
  int b = 0;
  nms(m_outputBbox, &prob[b * m_OUTPUT_SIZE], m_nms_threshold);
  std::cout << "number of detections -> " << prob[b * m_OUTPUT_SIZE] << std::endl;
  std::cout << "after nms -> " << m_outputBbox.size() << std::endl;
  cv::Mat tmp = img.clone();
  for (size_t j = 0; j < m_outputBbox.size(); j++) {
      if (m_outputBbox[j].class_confidence < m_bbox_threshold) continue;
      cv::Rect r = get_rect_adapt_landmark(tmp, m_INPUT_W, m_INPUT_H, m_outputBbox[j].bbox, m_outputBbox[j].landmark);
      cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
      //cv::putText(tmp, std::to_string((int)(m_outputBbox[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
      for (int k = 0; k < 10; k += 2) {
          cv::circle(tmp, cv::Point(m_outputBbox[j].landmark[k], m_outputBbox[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
      }
  }
  cv::imwrite(std::to_string(b) + "_result.jpg", tmp);
}