#include "retinaface.h"
#include "arcface.h"
#include "cosine_similarity.h"

#include <iostream>

static Logger gLogger;

int main(int argc, char **argv) {
    std::string face_engine_path = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/tensorrtx/retinaface/build/retina_r50.engine";
    uint batch_size = 1;

    RetinaFace* face = new RetinaFace(gLogger, face_engine_path, 1, 0.4, 0.75);
    std::string hiep_path = "/home/hiep/dev/test_face_recognition/IMG_0678.JPG";
    hiep_path = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/face-recognition/test/frames/00010.jpg";    
    cv::Mat img = cv::imread(hiep_path);
    std::cout << face->findFace(img).size() << std::endl;

    // std::string img1 = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/tensorrtx/arcface/joey0.ppm";
    // std::string img2 = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/tensorrtx/arcface/joey1.ppm";
    
    // std::string arcface_engine_path = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/tensorrtx/arcface/build/arcface-r100.engine";
    // ArcFaceR100* arcFace = new ArcFaceR100(gLogger, arcface_engine_path, 1);

    // // std::vector<decodeplugin::Detection>  bboxes = face->findFace(img);
    // // arcFace->extractFeature(img, bboxes);

    // cv::Mat img_data = cv::imread(img1);
    // float* prob = new float[512]; 
    // cv::resize(img_data, img_data, cv::Size(112, 112), 0, 0, cv::INTER_CUBIC);
    // arcFace->extractFeature(img_data, prob);
    // // std::cout << prob[0] << " " << prob[10] <<  std::endl;
    // cv::Mat out(512, 1, CV_32FC1, prob);
    // cv::Mat out_norm;
    // cv::normalize(out, out_norm);

    // cv::Mat img_data2 = cv::imread(img2);
    // float* prob1 = new float[512];
    // cv::resize(img_data2, img_data2, cv::Size(112, 112), 0, 0, cv::INTER_CUBIC);
    // arcFace->extractFeature(img_data2, prob1);
    // // std::cout << prob1[0] << " " << prob1[10] <<  std::endl;
    // cv::Mat out1(1, 512, CV_32FC1, prob1);
    // cv::Mat out_norm1;
    // cv::normalize(out1, out_norm1);

    // cv::Mat result = out_norm1 * out_norm;

    // std::cout << "similarity score: " << *(float*)result.data << std::endl;
    

    // test math mul
    // CosineSimilarityCalculator matmul;
    // matmul.init(prob, 1, 512);

    // float* m_outputs = new float[1 * 1];

    // matmul.calculate(prob1, 1, m_outputs);

    // for (int i = 0; i < 1; i++) {
    //     std::cout << m_outputs[i] << " ";
    // }
    // std::cout << std::endl;

    return 0;
}