#include "retinaface.h"

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

  checkCudaStatus(cudaStreamDestroy(stream));
  checkCudaStatus(cudaFree(io_buffers_[inputIndex]));
  checkCudaStatus(cudaFree(io_buffers_[outputIndex]));
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
  // cv::imwrite("preprocessed.jpg", pr_img);

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
  
  static int i = 0;
  int b = 0;
  nms(m_outputBbox, &prob[b * m_OUTPUT_SIZE], m_nms_threshold);
  // // std::cout << "number of detections -> " << prob[b * m_OUTPUT_SIZE] << std::endl;
  // std::cout << "after nms -> " << m_outputBbox.size() << std::endl;
  cv::Mat tmp = img.clone();
  for (size_t j = 0; j < m_outputBbox.size(); j++) {
      if (m_outputBbox[j].class_confidence < m_bbox_threshold) continue;
      cv::Rect r = get_rect_adapt_landmark(tmp, m_INPUT_W, m_INPUT_H, m_outputBbox[j].bbox, m_outputBbox[j].landmark);
      cv::Mat tempCrop = tmp(r);
      // uncomment if you want to save cropped face
      cv::imwrite(std::to_string(i) + "_" + std::to_string(j) + "_tempCrop.jpg", tempCrop);
      cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
      //cv::putText(tmp, std::to_string((int)(m_outputBbox[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
      for (int k = 0; k < 10; k += 2) {
          cv::circle(tmp, cv::Point(m_outputBbox[j].landmark[k], m_outputBbox[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
      }
  }
  
  // cv::imwrite(std::to_string(i) + "_result.jpg", tmp);
  i++;
}

inline cv::Mat RetinaFace::preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

inline cv::Rect RetinaFace::get_rect_adapt_landmark(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10]) {
    int l, r, t, b;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] / r_w;
        r = bbox[2] / r_w;
        t = (bbox[1] - (input_h - r_w * img.rows) / 2) / r_w;
        b = (bbox[3] - (input_h - r_w * img.rows) / 2) / r_w;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] /= r_w;
            lmk[i + 1] = (lmk[i + 1] - (input_h - r_w * img.rows) / 2) / r_w;
        }
    } else {
        l = (bbox[0] - (input_w - r_h * img.cols) / 2) / r_h;
        r = (bbox[2] - (input_w - r_h * img.cols) / 2) / r_h;
        t = bbox[1] / r_h;
        b = bbox[3] / r_h;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] = (lmk[i] - (input_w - r_h * img.cols) / 2) / r_h;
            lmk[i + 1] /= r_h;
        }
    }

    bbox[0] = l;
    bbox[1] = t;
    bbox[2] = r;
    bbox[3] = b;
    
    return cv::Rect(l, t, r-l, b-t);
}

float RetinaFace::iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0], rbox[0]), //left
        std::min(lbox[2], rbox[2]), //right
        std::max(lbox[1], rbox[1]), //top
        std::min(lbox[3], rbox[3]), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) -interBoxS + 0.000001f);
}

bool RetinaFace::cmp(const decodeplugin::Detection& a, const decodeplugin::Detection& b) {
    return a.class_confidence > b.class_confidence;
}

inline void RetinaFace::nms(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh) {
    std::vector<decodeplugin::Detection> dets;
    for (int i = 0; i < output[0]; i++) {
        if (output[15 * i + 1 + 4] <= 0.1) continue;
        decodeplugin::Detection det;
        memcpy(&det, &output[15 * i + 1], sizeof(decodeplugin::Detection));
        dets.push_back(det);
    }
    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t m = 0; m < dets.size(); ++m) {
        auto& item = dets[m];
        res.push_back(item);
        //std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
        for (size_t n = m + 1; n < dets.size(); ++n) {
            if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                dets.erase(dets.begin()+n);
                --n;
            }
        }
    }
}


