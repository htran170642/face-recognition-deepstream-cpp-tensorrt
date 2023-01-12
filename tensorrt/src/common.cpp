#include "common.h"

bool fileExists(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void getFilePaths(std::string rootPath, std::vector<struct Paths> &paths, std::set<std::string> &userIds) {
    /*
    imagesPath--|
                |--class0--|
                |          |--f0.jpg
                |          |--f1.jpg
                |
                |--class1--|
                           |--f0.jpg
                           |--f1.jpg
    ...
    */
    DIR *dir;
    struct dirent *entry;
    std::string postfix = ".mp4"; // exclude .mp4, .mov
    if ((dir = opendir(rootPath.c_str())) != NULL) {
        try {
            while (((entry = readdir(dir)) != NULL) && entry->d_type == DT_DIR) {
                try {
                    std::string class_path = rootPath + "/" + entry->d_name;
                    std::cout << "class_path :" << class_path << std::endl;
                    DIR *class_dir = opendir(class_path.c_str());
                    struct dirent *file_entry;
                    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
                    while ((file_entry = readdir(class_dir)) != NULL) {
                        std::string name(file_entry->d_name);
                        if (strcmp(name.c_str(), ".") != 0 &&
                            strcmp(name.c_str(), "..") != 0) {
                            if (file_entry->d_type != DT_DIR 
                                    && 0 != name.compare(name.length() - postfix.length(), postfix.length(), postfix)) {
                                struct Paths tempPaths;
                                tempPaths.className = std::string(entry->d_name);
                                tempPaths.absPath = class_path + "/" + name;
                                paths.push_back(tempPaths);
                                userIds.insert(tempPaths.className);
                            }
                        }
                    }
                } catch (const char *s) {
                    std::cout << "Exception: " << s;
                } 
                
            }
        } catch (const char *s) {
                std::cout << "Exception: " << s;
        } 
        closedir(dir);
    }
}

void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA API failed with status " << status << ": " << "cudaGetErrorString(status)" << std::endl;
        throw std::logic_error("CUDA API failed");
    }
}

void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS API failed with status " << status << std::endl;
        throw std::logic_error("cuBLAS API failed");
    }
}

// static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
//     DIR *p_dir = opendir(p_dir_name);
//     if (p_dir == nullptr) {
//         return -1;
//     }

//     struct dirent* p_file = nullptr;
//     while ((p_file = readdir(p_dir)) != nullptr) {
//         if (strcmp(p_file->d_name, ".") != 0 &&
//             strcmp(p_file->d_name, "..") != 0) {
//             //std::string cur_file_name(p_dir_name);
//             //cur_file_name += "/";
//             //cur_file_name += p_file->d_name;
//             std::string cur_file_name(p_file->d_name);
//             file_names.push_back(cur_file_name);
//         }
//     }

//     closedir(p_dir);
//     return 0;
// }