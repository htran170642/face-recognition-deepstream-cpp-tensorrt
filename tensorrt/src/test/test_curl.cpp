#include <curl/curl.h>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

static size_t WriteMemoryCallback(char *contents, size_t size, size_t nmemb, void *userp){
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
  }

int main(int argc, char **argv) {

    cv::Mat frame;
    std::string path = "/home/hiep/dev/test_face_recognition/hiep2.jpg";
    path = "/home/hiep/dev/test_face_recognition/hiep0_tempCrop.jpg";
    frame = cv::imread(path.c_str());
    std::cout << "image size:" << frame.size() <<std::endl;

    // Encode
    std::vector<uchar> buf;
    cv::imencode(".jpg", frame, buf, std::vector<int>{cv::IMWRITE_JPEG_QUALITY, 100});
    std::string imgStrBuff = std::string(buf.begin(), buf.end());
    char *imgBuffPtr = (char *)&imgStrBuff;
    long imgBuffLength = static_cast<long>(imgStrBuff.size());
    std::string readBuffer;
   
    std::string url = "http://0.0.0.0:18080/recognize";

    CURL *curl;
    CURLcode res;

    /* In windows, this will init the winsock stuff */
    curl_global_init(CURL_GLOBAL_ALL);

    /* get a curl handle */
    curl = curl_easy_init();

    if(curl) {
        struct curl_slist *headers;
        headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, imgStrBuff.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, imgBuffLength);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "curl/7.38.0");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 50L);
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK)
            fprintf(stderr, "curl_easy_perform() failed: %s\n",
                curl_easy_strerror(res));

        curl_easy_cleanup(curl);
        std::cout << "result: " <<readBuffer << std::endl;
    }

    return 0;
}