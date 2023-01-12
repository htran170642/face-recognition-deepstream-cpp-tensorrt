#ifndef COMMON_H
#define COMMON_H

#include "cuda_runtime_api.h"
#include <cublasLt.h>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include "driver_types.h"
#include <cuda.h>

#include <opencv2/opencv.hpp>
#include <dirent.h>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

struct Paths {
    std::string absPath;
    std::string className;
};

bool fileExists(const std::string &name);
void getFilePaths(std::string rootPath, std::vector<struct Paths> &paths, std::set<std::string> &userIds);
void checkCudaStatus(cudaError_t status);
void checkCublasStatus(cublasStatus_t status);

// static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);

#endif // COMMON_H