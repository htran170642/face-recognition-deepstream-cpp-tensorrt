#include "retinaface.h"

static Logger gLogger;

int main(int argc, char **argv) {
    std::string face_engine_path = "/home/hiep/dev/test_face_recognition_tensorrt_deepstream/tensorrtx/retinaface/build/retina_r50.engine";
    uint batch_size = 1;

    RetinaFace* face = new RetinaFace(gLogger, face_engine_path, 1, 0.4, 0.75);
    cv::Mat img = cv::imread("../worlds-largest-selfie.jpg");
    std::cout << face->findFace(img).size() << std::endl;
    return 0;
}