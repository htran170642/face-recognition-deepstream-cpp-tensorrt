/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
// #include "decode.h"
#include <map>

namespace decodeplugin
{
    struct alignas(float) Detection{
        float bbox[4];  //x1 y1 x2 y2
        float class_confidence;
        float landmark[10];
    };
    static const int INPUT_H = 480;
    static const int INPUT_W = 640;
}

static const int INPUT_H = decodeplugin::INPUT_H;  // H, W must be able to  be divided by 32.
static const int INPUT_W = decodeplugin::INPUT_W;
static const int N_ANCHORS = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2;

#define kNMS_THRESH 0.45

static float iou(float lbox[4], float rbox[4]) {
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

static bool cmp(const decodeplugin::Detection& a, const decodeplugin::Detection& b) {
    return a.class_confidence > b.class_confidence;
}

static inline void nms(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh = 0.4, float conf_thresh = 0.5) {
    std::vector<decodeplugin::Detection> dets;
    for (int i = 0; i < output[0]; i++) {
        if (output[15 * i + 1 + 4] <= conf_thresh) continue;
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

/* This is a sample bounding box parsing function for the sample YoloV5 detector model */
static bool NvDsInferParseRetinaface(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    const float kCONF_THRESH = detectionParams.perClassThreshold[0];

    std::vector<decodeplugin::Detection> res;
    nms(res, (float*)(outputLayersInfo[0].buffer), kNMS_THRESH, kCONF_THRESH);

    for(auto& r : res) {
	    NvDsInferParseObjectInfo oinfo;        
        
	    oinfo.classId = 0;
	    oinfo.left    = static_cast<unsigned int>(r.bbox[0]);
	    oinfo.top     = static_cast<unsigned int>(r.bbox[1]);
	    oinfo.width   = static_cast<unsigned int>(r.bbox[2] - r.bbox[0]);
	    oinfo.height  = static_cast<unsigned int>(r.bbox[3] - r.bbox[1]);
	    oinfo.detectionConfidence = r.class_confidence;
	    objectList.push_back(oinfo);        
    }
    
    return true;
}

extern "C" bool NvDsInferParseCustomRetinaface(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseRetinaface(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomRetinaface);
