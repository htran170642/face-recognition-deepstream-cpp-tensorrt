#include "arcface.h"

void getCroppedFaces(cv::Mat frame, std::vector<decodeplugin::Detection> &outputBbox, std::vector<struct CroppedFace> &croppedFaces) {
  croppedFaces.clear();
  // cv::imwrite("frame.jpg", frame);
  for (auto &out : outputBbox) {
        cv::Rect facePos(cv::Point(out.bbox[0], out.bbox[1]), cv::Point(out.bbox[2], out.bbox[3]));
        cv::Mat tempCrop = frame(facePos);
        // cv::imwrite("tempCrop.jpg", frame);
        struct CroppedFace currFace;
        cv::resize(tempCrop, currFace.faceMat, cv::Size(REG_INPUT_HEIGHT, REG_INPUT_WIDTH), 0, 0, cv::INTER_CUBIC); // resize to network dimension input
        currFace.face = currFace.faceMat.clone();
        currFace.x1 = out.bbox[0];
        currFace.y1 = out.bbox[1];
        currFace.x2 = out.bbox[2];
        currFace.y2 = out.bbox[3];
        croppedFaces.push_back(currFace);

        // static int i = 0;
        // cv::imwrite(std::to_string(i)+".jpg", currFace.face);
        // i++;
    }
}

int ArcFaceR100::classCount = 0;

ArcFaceR100::ArcFaceR100(Logger gLogger, const std::string engineFile, int maxBatchSize) {
  
  m_INPUT_H = REG_INPUT_HEIGHT;
  m_INPUT_W = REG_INPUT_WIDTH;
  m_INPUT_SIZE  = 3 * m_INPUT_H * m_INPUT_W;
  m_OUTPUT_SIZE = 512;
  m_maxBatchSize = maxBatchSize;
  m_output = new float[m_maxBatchSize * m_OUTPUT_SIZE];

  m_embed = new float[m_OUTPUT_SIZE];
  croppedFaces.reserve(MAX_FACES_PER_IMAGE);
  m_embeds = new float[MAX_FACES_PER_IMAGE * m_OUTPUT_SIZE];

  // load engine from .engine file
  loadEngine(gLogger, engineFile);
}

ArcFaceR100::~ArcFaceR100() {
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

void ArcFaceR100::loadEngine(Logger gLogger, const std::string engineFile) {

  if (fileExists(engineFile)) {
    std::cout << "[INFO] Loading Arcface Engine...\n";
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

void ArcFaceR100::preprocessFace(cv::Mat &face) {
  m_input.release();
  cv::resize(face, face, cv::Size(REG_INPUT_HEIGHT, REG_INPUT_WIDTH), 0, 0, cv::INTER_CUBIC);
  cv::cvtColor(face, face, cv::COLOR_BGR2RGB);
  face.convertTo(face, CV_32F);
  face = (face - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
  std::vector<cv::Mat> temp;
  cv::split(face, temp);
  for (int i = 0; i < temp.size(); i++) {
      m_input.push_back(temp[i]);
  }
}

void ArcFaceR100::preprocessFaces() {
  for (int i = 0; i < croppedFaces.size(); i++) {
    cv::cvtColor(croppedFaces[i].faceMat, croppedFaces[i].faceMat, cv::COLOR_BGR2RGB);
    croppedFaces[i].faceMat.convertTo(croppedFaces[i].faceMat, CV_32F);
    croppedFaces[i].faceMat = (croppedFaces[i].faceMat - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
    std::vector<cv::Mat> temp;
    cv::split(croppedFaces[i].faceMat, temp);
    for (int i = 0; i < temp.size(); i++) {
        m_input.push_back(temp[i]);
    }
    croppedFaces[i].faceMat = m_input.clone();
    m_input.release();
  }
}

void ArcFaceR100::inference(float* data, float* prob) {
  checkCudaStatus(cudaMemcpyAsync(io_buffers_[inputIndex], data, m_INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
  m_context_->enqueueV2(io_buffers_, stream, nullptr);
  checkCudaStatus(cudaMemcpyAsync(prob, io_buffers_[outputIndex], m_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void ArcFaceR100::inference(float *input, float *output, int batchSize) {
    // Set input dimensions
    m_context_->setBindingDimensions(inputIndex, nvinfer1::Dims4(batchSize, 3, m_INPUT_H, m_INPUT_W));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    checkCudaStatus(cudaMemcpyAsync(io_buffers_[inputIndex], input, batchSize * m_INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
    m_context_->enqueueV2(io_buffers_, stream, nullptr);
    checkCudaStatus(cudaMemcpyAsync(output, io_buffers_[outputIndex], batchSize * m_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

// model output arcfacer100 is not normalized yet. So the mathmul calcualtion is greater than 1 (depends on model output)
void ArcFaceR100::normalize(float* output) {
  cv::Mat out(1, m_OUTPUT_SIZE, CV_32FC1, output);
  cv::Mat out_norm;
  cv::normalize(out, out_norm);
  memcpy(output, (float *)out_norm.ptr<float>(0), 1 * m_OUTPUT_SIZE * sizeof(float));
}

void ArcFaceR100::extractFeature(cv::Mat &img, float *m_output) {
  preprocessFace(img);
  inference((float *)m_input.ptr<float>(0), m_output);
  // normalize(m_output);
}

void ArcFaceR100::extractFeature(cv::Mat frame, std::vector<decodeplugin::Detection> &outputBbox) {
  getCroppedFaces(frame, outputBbox, croppedFaces);
  preprocessFaces();
  if (m_maxBatchSize < 2) {
    for(int i=0; i < croppedFaces.size(); i++) { 
      inference((float *)croppedFaces[i].faceMat.ptr<float>(0), m_embed);
      // normalize(m_embed);
      std::copy(m_embed, m_embed + m_OUTPUT_SIZE, m_embeds  + i * m_OUTPUT_SIZE);
    }
  } else {
    int num = croppedFaces.size();
    int end = 0;
    for (int beg = 0; beg < croppedFaces.size(); beg = beg + m_maxBatchSize) {
        end = std::min(num, beg + m_maxBatchSize);
        cv::Mat input;
        for (int i = beg; i < end; ++i) {
            input.push_back(croppedFaces[i].faceMat);
        }
        inference((float *)input.ptr<float>(0), m_embed, end - beg);
        // normalize(m_embed);
        std::copy(m_embed, m_embed + (end - beg) * m_OUTPUT_SIZE, m_embeds + (end - beg) * beg * m_OUTPUT_SIZE);
    }
  }
}

float *ArcFaceR100::matchFeature() {
    /*
        Get cosine similarity matrix of known embeddings and new embeddings.
        Since output is l2-normed already, only need to perform matrix multiplication.
    */
    m_outputs = new float[croppedFaces.size() * classCount];
    if (classNames.size() > 0 && croppedFaces.size() > 0) {
        matmul.calculate(m_embeds, croppedFaces.size(), m_outputs);
    } else {
        throw "Feature matching: No faces in database or no faces found";
    }

    // print prob
    // for (int i = 0; i < croppedFaces.size()*classCount; i++) {
    //     std::cout << m_outputs[i] << " ";
    // }
    // std::cout << std::endl;
    
    return m_outputs;
}

std::tuple<std::vector<std::string>, std::vector<float>> ArcFaceR100::getOutputs(float *output_sims) {
    /*
        Get person corresponding to maximum similarity score based on cosine similarity matrix.
    */
    std::vector<std::string> names;
    std::vector<float> sims;
    for (int i = 0; i < croppedFaces.size(); ++i) {
        int argmax = std::distance(output_sims + i * classCount, std::max_element(output_sims + i * classCount, output_sims + (i + 1) * classCount));
        float sim = *(output_sims + i * classCount + argmax);
        std::string name = classNames[argmax];
        names.push_back(name);
        sims.push_back(sim);
    }
    return std::make_tuple(names, sims);
}

void ArcFaceR100::addEmbedding(const std::string className, float embedding[]) {
    classNames.push_back(className);
    std::copy(embedding, embedding + m_OUTPUT_SIZE, m_knownEmbeds + classCount * m_OUTPUT_SIZE);
    classCount++;
}

void ArcFaceR100::addEmbedding(const std::string className, std::vector<float> embedding) {
    classNames.push_back(className);
    std::copy(embedding.begin(), embedding.end(), m_knownEmbeds + classCount * m_OUTPUT_SIZE);
    classCount++;
}
void ArcFaceR100::initKnownEmbeds(int num) { m_knownEmbeds = new float[num * m_OUTPUT_SIZE]; }
void ArcFaceR100::initMatMul() { 
    matmul.init(m_knownEmbeds, classCount, m_OUTPUT_SIZE); 
}