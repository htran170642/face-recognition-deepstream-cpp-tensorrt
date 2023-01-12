// #include "retinaface.h"
#include "arcface.h"
#include "cosine_similarity.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>
#include "webclient.h"

using json = nlohmann::json;

static Logger gLogger;

int main(int argc, char **argv) {
    // std::string face_engine_path = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/tensorrtx/retinaface/build/retina_r50.engine";
    // uint batch_size = 1;

    // RetinaFace* face = new RetinaFace(gLogger, face_engine_path, 1, 0.4, 0.75);
    // cv::Mat img = cv::imread("/home/hiep/dev/test_face_recognition/IMG_0678.JPG");
    // std::cout << face->findFace(img).size() << std::endl;

    // std::string img1 = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/tensorrtx/arcface/joey0.ppm";
    // std::string img2 = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/tensorrtx/arcface/joey1.ppm";
    
    // std::string arcface_engine_path = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/tensorrtx/arcface/build/arcface-r100.engine";
    // arcface_engine_path = "/home/hiep/dev/test_face_recognition/face-recognition-cpp-tensorrt/conversion/arcface/arcface-ir50_asia-112x112-b1-fp16.engine";
    // ArcFaceR100* arcFace = new ArcFaceR100(gLogger, arcface_engine_path, 1);

    // // // std::vector<decodeplugin::Detection>  bboxes = face->findFace(img);
    // // // arcFace->extractFeature(img, bboxes);

    // cv::Mat img_data = cv::imread(img1);
    // float* prob = new float[512]; 
    // arcFace->extractFeature(img_data, prob);
    // // // std::cout << prob[0] << " " << prob[10] <<  std::endl;
    // cv::Mat out(512, 1, CV_32FC1, prob);
    // // cv::Mat out_norm;
    // // cv::normalize(out, out_norm);

    // cv::Mat img_data2 = cv::imread(img2);
    // float* prob1 = new float[512];
    // arcFace->extractFeature(img_data2, prob1);
    // // std::cout << prob1[0] << " " << prob1[10] <<  std::endl;
    // cv::Mat out1(1, 512, CV_32FC1, prob1);
    // // cv::Mat out_norm1;
    // // cv::normalize(out1, out_norm1);

    // // cv::Mat result = out_norm1 * out_norm;
    // cv::Mat result = out1 * out;
    // std::cout << "similarity score: " << *(float*)result.data << std::endl;


    // // // test math mul
    // CosineSimilarityCalculator matmul;
    // matmul.init(prob, 1, 512);

    // float* m_outputs = new float[1 * 1];

    // matmul.calculate(prob1, 1, m_outputs);

    // for (int i = 0; i < 1; i++) {
    //     std::cout << m_outputs[i] << " ";
    // }
    // std::cout << std::endl;


    // HTTP request client
    std::string host = "localhost";
    std::string port = "18080";
    std::string url = "/recognize";
    HttpClient http(host, port, url);

    // Variables for response
    json j;
    std::string res;

    // Read image
    cv::Mat frame;
    std::string path = "/home/hiep/dev/test_face_recognition/hiep2.jpg";
    path = "/home/hiep/dev/test_face_recognition/hiep0_tempCrop.jpg";
    frame = cv::imread(path.c_str());

    // Encode
    std::vector<uchar> buf;
    cv::imencode(".jpg", frame, buf, std::vector<int>{cv::IMWRITE_JPEG_QUALITY, 100});
    std::string s = std::string(buf.begin(), buf.end());

    // Send
    res = http.send(s);

    // Buffer to json
    j = json::parse(res);
    if (j.size() > 0) {
        std::cout << "Prediction: " << j["userId"] << " " << j["similarity"] << "\n";
    } else {
        std::cout << "No prediction\n";
    }

    return 0;
}