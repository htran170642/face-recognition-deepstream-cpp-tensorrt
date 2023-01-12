#include "db.h"
#include "crow.h"
#include "retinaface.h"
#include "arcface.h"
#include "db.h"
#include <nlohmann/json.hpp>
#include <boost/lexical_cast.hpp>
#include <chrono>

using json = nlohmann::json;

static Logger gLogger;

template<typename F>
constexpr auto timing(const F& func) {
    return [func](auto&&... args) {
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>
              (std::chrono::system_clock::now().time_since_epoch()).count();
        func(std::forward<decltype(args)>(args)...);
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>
              (std::chrono::system_clock::now().time_since_epoch()).count();
        CROW_LOG_INFO << "time: "<< end - start<< "ms";
    };
}

int main(int argc, char **argv) {

    // Config
    CROW_LOG_INFO << "Loading config...";
    std::string configPath = "../config.json";
    if (argc < 2 || (strcmp(argv[1], "-c") != 0)) {
        CROW_LOG_INFO << "Please specify config file path with -c option. Use default path: \"" << configPath << "\"";
    } else {
        configPath = argv[2];
        CROW_LOG_INFO << "Config path: \"" << configPath << "\"";
    }
    std::ifstream configStream(configPath);
    json config;
    configStream >> config;
    configStream.close();

    // params
    std::string databasePath = config["database_path"];
    std::string detEngineFile = config["det_engine"];
    std::vector<int> detInputShape = config["det_inputShape"];
    std::string detInputName = config["det_inputName"];
    std::string detOutputName = config["det_outputName"];
    int detMaxBatchSize = config["det_maxBatchSize"];
    float det_threshold_nms = config["det_threshold_nms"];
    float det_threshold_bbox = config["det_threshold_bbox"];
    std::vector<int> recInputShape = config["rec_inputShape"];
    int recOutputDim = config["rec_outputDim"];
    std::string recEngineFile = config["rec_engine"];
    int maxFacesPerScene = config["det_maxFacesPerScene"];
    float knownPersonThreshold = config["rec_knownPersonThreshold"];
    int recMaxBatchSize = config["rec_maxBatchSize"];
    std::string recInputName = config["rec_inputName"];
    std::string recOutputName = config["rec_outputName"];
    bool apiImageIsCropped = config["api_imgIsCropped"];

    RetinaFace detector(gLogger, detEngineFile, detMaxBatchSize, det_threshold_nms, det_threshold_bbox);
    ArcFaceR100 recognizer(gLogger, recEngineFile, recMaxBatchSize);

    // init db
    Database db = Database(databasePath, recOutputDim);
    // dict that map USR_ID to USR_NM
    std::map<std::string, std::string> userDict = db.getUserDict();

    // init bbox and allocate memory according to maxFacesPerScene
    std::vector<struct decodeplugin::Detection> outputBbox;
    outputBbox.reserve(MAX_FACES_PER_IMAGE);

    if (config["gen_database"]) { 
        // generate db
        auto total = 0;
        CROW_LOG_INFO << "Parsing images from " << config["gen_imgSource"];
        std::vector<struct Paths> paths;
        std::set<std::string> userIds;
        getFilePaths(config["gen_imgSource"], paths, userIds);
        for (auto userId : userIds) {
            db.insertUser(userId, userId);
        }

        cv::Mat image;
        float output[recOutputDim];
        // create database from folder
        CROW_LOG_INFO << "Creating database...";
        for (int i = 0; i < paths.size(); i++) {
            auto start = std::chrono::duration_cast<std::chrono::milliseconds>
                        (std::chrono::system_clock::now().time_since_epoch()).count();;
            try {
                std::string userId = paths[i].className;
                image = cv::imread(paths[i].absPath.c_str());
                CROW_LOG_INFO << "UserID: " << userId;
                CROW_LOG_INFO << "Image: " << image.size();
                CROW_LOG_INFO << "Finding faces in image...";
                outputBbox = detector.findFace(image);

                std::vector<struct CroppedFace> croppedFaces;
                getCroppedFaces(image, outputBbox, croppedFaces);
                CROW_LOG_INFO << "There are " << croppedFaces.size() << " face(s) in image.";

                if (croppedFaces.size() == 1) {
                    CROW_LOG_INFO << "Getting embedding...";
                    recognizer.extractFeature(croppedFaces[0].faceMat, output);
                    db.insertFace(userId, paths[i].absPath, output);
                }
                outputBbox.clear();
            } catch (const char *s) {
                CROW_LOG_WARNING << "Exception: " << s;
            } 
            auto end = std::chrono::duration_cast<std::chrono::milliseconds>
              (std::chrono::system_clock::now().time_since_epoch()).count();
            total += end - start;
            CROW_LOG_INFO << "time: "<< end - start << "ms";
        }
        CROW_LOG_INFO << "total time: "<< total << "ms";
        CROW_LOG_INFO << "total embeddings: "<< paths.size() << " ids";
        CROW_LOG_INFO << "Database generated. Exitting...";
        db.exportToJson();
        CROW_LOG_INFO << "Embedding json generated. Exitting...";
        
        exit(0);
    } else {
        // load from database
        CROW_LOG_INFO << "Reading embeddings from database...";
        db.getEmbeddings(recognizer);
        CROW_LOG_INFO << "Init cuBLASLt matrix multiplication class...";
        recognizer.initMatMul();
    }    
    
    // init opencv and output vectors
    cv::Mat rawInput;
    cv::Mat frame;
    float *output_sims;
    std::vector<std::string> names;
    std::vector<float> sims;

    crow::SimpleApp app;

    CROW_ROUTE(app, "/health")([](){
        return "OK";
    });

    CROW_ROUTE(app, "/insert/user")
    .methods(crow::HTTPMethod::Post)([&db](const crow::request& req) {
        auto x = crow::json::load(req.body);
        if (!x)
            return crow::response(crow::status::BAD_REQUEST);
        std::string userId = x["userId"].s();
        std::string userName = x["userName"].s();
        int ret = db.insertUser(userId, userName);
        std::string response = "Fail! User `" + userId + "` already in database.\n";
        if (ret == 1)
            response = "Success! User `" + userId + "` inserted.\n";
        return crow::response(response);
    });

    CROW_ROUTE(app, "/insert/face")
    .methods(crow::HTTPMethod::Post)([&](const crow::request& req) {
        json j;
        std::string response = "";
        std::string info;
        int ret = 0;

        try {
            j = json::parse(req.body);
            if (j.contains("data")) {
                for (auto &el : j["data"].items()) {
                    std::string userId = el.value()["userId"];
                    std::string imgPath = el.value()["imgPath"];
                    if (!fileExists(imgPath))
                        throw "Image path not found";

                    cv::Mat image = cv::imread(imgPath.c_str());
                    float output[512];

                    if(apiImageIsCropped) {
                        CROW_LOG_INFO << "Image: " << image.size();
                        if (image.empty())
                            throw "Empty image";
                        CROW_LOG_INFO << "Getting embedding...";
                        recognizer.extractFeature(image, output);
                        ret = 1;
                    } else {
                        CROW_LOG_INFO << "Image: " << image.size();
                        CROW_LOG_INFO << "Finding faces in image...";
                        outputBbox = detector.findFace(image);

                        std::vector<struct CroppedFace> croppedFaces;
                        getCroppedFaces(image, outputBbox, croppedFaces);
                        CROW_LOG_INFO << "There are " << croppedFaces.size() << " face(s) in image.";

                        if (croppedFaces.size() > 1) {
                            response += "There are more than 1 faces in input image from `" + imgPath + "`\n";
                            ret = 2;
                        } else if (croppedFaces.size() == 0) {
                            response += "Cant find any faces in input image from `" + imgPath + "`\n";
                            ret = 3;
                        } else {
                            CROW_LOG_INFO << "Getting embedding...";
                            response += "1 face found in input image from `" + imgPath + "`, processing...\n";
                            recognizer.extractFeature(croppedFaces[0].faceMat, output);
                            ret = 1;
                        }                   
                    }
                    if (ret != 1) {
                        response += "Fail! Embedding for `" + userId + "` cannot be inserted.\n";
                    } else {
                        ret = db.insertFace(userId, imgPath, output);
                        if (ret == 1) {
                            response += "Success! Embedding for `" + userId + "` inserted successfully.\n";
                        } else {
                            response += "Fail! Embedding for `" + userId + "` cannot be inserted.\n";
                        }
                    }
                    // clean
                    outputBbox.clear();
                    image.release();
                }
                
            } else {
                response = "Cant find field `data` in input!\n";
            }
        } catch (json::parse_error &e) {
            CROW_LOG_ERROR << "JSON parsing error: " << e.what() << '\n' << "exception id: " << e.id;
            response = "Please check json input\n";
        } catch (const char *s) {
            CROW_LOG_WARNING << "Exception: " << s;
            response = s;
            response += "\n";
        }
        return crow::response(response);
    });

    CROW_ROUTE(app, "/delete/user")
    ([&db](const crow::request &req) {
        if (req.url_params.get("id") == nullptr)
            return crow::response("Failed\n");
        else {
            std::string userId = req.url_params.get("id");
            db.deleteUser(userId);
        }

        return crow::response("Success\n");
    });

    CROW_ROUTE(app, "/delete/face")
    ([&db](const crow::request &req) {
        if (req.url_params.get("id") == nullptr)
            return crow::response("Failed\n");
        else {
            int id = boost::lexical_cast<int>(req.url_params.get("id"));
            db.deleteFace(id);
        }

        return crow::response("Success\n");
    });
    
    /*
    * input is a cropped face image
    */
    CROW_ROUTE(app, "/recognize").methods(crow::HTTPMethod::Post)([&](const crow::request &req) {
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>
              (std::chrono::system_clock::now().time_since_epoch()).count();
        crow::json::wvalue retval;
        try {
            std::string decoded = req.body;
            std::vector<uchar> data(decoded.begin(), decoded.end());
            frame = cv::imdecode(data, cv::IMREAD_UNCHANGED);
            // cv::imwrite("encode2.jpg", frame);
            CROW_LOG_INFO << "Image: " << frame.size();
            if (frame.empty())
                throw "Empty image";
            int height = frame.size[0];
            int width = frame.size[1];
            // resize if diff size
            if ((height != REG_INPUT_HEIGHT) || (width != REG_INPUT_WIDTH)) {
                CROW_LOG_INFO << "Resizing input to " << REG_INPUT_HEIGHT << "x" << REG_INPUT_WIDTH;
                cv::resize(frame, frame, cv::Size(REG_INPUT_HEIGHT, REG_INPUT_WIDTH));
            }
            CROW_LOG_INFO << "Getting embedding...";
            decodeplugin::Detection bbox; 
            bbox.bbox[0] = 0;
            bbox.bbox[1] = 0;
            bbox.bbox[2] = REG_INPUT_HEIGHT;
            bbox.bbox[3] = REG_INPUT_WIDTH;
            bbox.class_confidence = 1;
            outputBbox.push_back(bbox);
            recognizer.extractFeature(frame, outputBbox);
            CROW_LOG_INFO << "Feature matching...";
            output_sims = recognizer.matchFeature();
            std::tie(names, sims) = recognizer.getOutputs(output_sims);
            retval = {
                {"userId", sims[0] > knownPersonThreshold ? names[0] : "unknown" },
                {"similarity", sims[0]},
            };
            CROW_LOG_INFO << "Prediction: " << names[0] << " " << sims[0];
            auto end = std::chrono::duration_cast<std::chrono::milliseconds>
              (std::chrono::system_clock::now().time_since_epoch()).count();
            CROW_LOG_INFO << "time: "<< end - start << "ms";
        } catch (const char *s) {
            CROW_LOG_WARNING << "Exception: " << s;
        }

        // clean
        outputBbox.clear();
        names.clear();
        sims.clear();
        frame.release();

        return crow::response(retval);
    });

    // ignore all log
    crow::logger::setLogLevel(crow::LogLevel::Info);
    //crow::logger::setHandler(std::make_shared<ExampleLogHandler>());

    app.port(18080)
      .server_name("Face")
      .multithreaded()
      .run();

    return 0;
}