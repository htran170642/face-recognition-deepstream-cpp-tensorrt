#include "retinaface.h"
#include "arcface.h"
#include "cosine_similarity.h"

#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static Logger gLogger;

int main(int argc, char **argv) {
    // std::string face_engine_path = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/tensorrtx/retinaface/build/retina_r50.engine";
    // uint batch_size = 1;

    // RetinaFace* face = new RetinaFace(gLogger, face_engine_path, 1, 0.4, 0.75);
    // std::string hiep_path = "/home/hiep/dev/test_face_recognition/IMG_0678.JPG";
    // hiep_path = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/face-recognition/test/frames/00010.jpg";    
    // cv::Mat img = cv::imread(hiep_path);
    // std::cout << face->findFace(img).size() << std::endl;

    std::string img1 = "/home/hiep/dev/test_face_recognition/hiep0_tempCrop.jpg";
    img1 = "/home/hiep/dev/test_face_recognition/0_0_tempCrop.jpg";
    // img1 = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/face-recognition/deepstream/build/0_1.jpg";
    // img1 = "/home/hiep/dev/test_face_recognition/34_0_tempCrop.jpg";
    std::string arcface_engine_path = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/tensorrtx/arcface/build/arcface-r100.engine";
    arcface_engine_path = "/home/hiep/dev/test_face_recognition/face-recognition-cpp-tensorrt/conversion/arcface/arcface-ir50_asia-112x112-b1-fp16.engine";
    ArcFaceR100* arcFace = new ArcFaceR100(gLogger, arcface_engine_path, 1);

    // // std::vector<decodeplugin::Detection>  bboxes = face->findFace(img);
    // // arcFace->extractFeature(img, bboxes);

    cv::Mat img_data = cv::imread(img1);
    float* prob = new float[512]; 
    arcFace->extractFeature(img_data, prob);
    printf("%f %f %f %f %f % f %f %f %f %f\n", 
                            prob[0], prob[1], prob[2],
                            prob[3], prob[4], prob[5],
                            prob[6], prob[7], prob[8],
                            prob[9]);

    img1 = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/face-recognition/deepstream/build/0_0.jpg";
    float* prob1 = new float[512]; 
    cv::Mat img_data1 = cv::imread(img1);
    arcFace->extractFeature(img_data1, prob1);
    printf("%f %f %f %f %f % f %f %f %f %f\n", 
                            prob1[0], prob1[1], prob1[2],
                            prob1[3], prob1[4], prob1[5],
                            prob1[6], prob1[7], prob1[8],
                            prob1[9]);
    printf("\n");
    
    // test load embedding from json file
    std::ifstream is("../../../deepstream/config.json");
    json config;
    is >> config;
    is.close();
    is.clear();
    is.open(config["input_numImagesFile"]);
    std::string numImages_str;
    std::getline(is, numImages_str);
    unsigned int numImages = std::stoi(numImages_str);
    printf("[INFO] Num faces: %d\n", numImages);
    is.close();
    is.clear();
    printf("[INFO] Reading embeddings from file...\n");
    is.open(config["input_embeddingsFile"]);
    json j;
    is >> j;
    is.close();

    int outputDim = config["rec_outputDim"];
    std::vector<std::string> knownIds;
    float *knownEmbeds = new float[numImages * outputDim];
    unsigned int knownEmbedCount = 0;

    for (json::iterator it = j.begin(); it != j.end(); ++it){
        std::string id = it.key();
        std::cout << "Member: " << id << ", Num of images: " <<  it.value().size() <<'\n';
        for (int i = 0; i < it.value().size(); ++i) {
            knownIds.push_back(it.key());
            // std::cout << it.value()[i][0] << " " << it.value()[i][10] << '\n';
            std::copy(it.value()[i].begin(), it.value()[i].end(), knownEmbeds + knownEmbedCount * outputDim);
            knownEmbedCount++;
        }
    }
    
    printf("knownEmbedCount= %d\n", knownEmbedCount);
    // test math mul
    CosineSimilarityCalculator matmul;
    matmul.init(knownEmbeds, knownEmbedCount, 512);

    float* sims = new float[knownEmbedCount * 1];

    matmul.calculate(prob, 1, sims);

    for (int i = 0; i < knownEmbedCount*1; i++) {
        std::cout << sims[i] << " ";
    }
    std::cout << std::endl;

    int i = 0;
    int argmax = std::distance(
            sims + i * knownEmbedCount,
            std::max_element(sims + i * knownEmbedCount, sims + (i + 1) * knownEmbedCount));
    float sim = *(sims + i * knownEmbedCount + argmax);
    std::cout << "name: " << knownIds[argmax] << ", sim= " << sim << std::endl;
    // ex. name: hiep, sim= 0.72712
    return 0;
}