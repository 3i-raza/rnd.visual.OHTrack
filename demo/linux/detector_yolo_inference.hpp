//
//  detector_yolo_inference.hpp
//  Inference
//
//  Created by Song on 12/5/24.
//

#ifndef detector_yolo_inference_hpp
#define detector_yolo_inference_hpp

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include "layer.h"
#include "net.h"

#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <stdio.h>
#include <cstdio>
#include "boxinfo.h"
#include "detector_class_info.h"

#define MAX_STRIDE 32

#define RIDER_ID 0
#define HORSE_ID 17

#define PROB_THRESHOLD 0.5

static const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};


namespace yolo {

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

float intersection_area(const yolo::Object& a, const yolo::Object& b);

void qsort_descent_inplace(std::vector<yolo::Object>& objects, int left, int right);

void qsort_descent_inplace(std::vector<yolo::Object>& objects);

void nms_sorted_bboxes(const std::vector<yolo::Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false);

float sigmoid(float x);

float clampf(float d, float min, float max);

void parse_yolov_detections(
    float* inputs, float confidence_threshold,
    int num_channels, int num_anchors, int num_labels,
    int infer_img_width, int infer_img_height,
                                   std::vector<yolo::Object>& objects);

void detect(const DetectorClassInfo classInfo,
            ncnn::Net& yoloModel, const cv::Mat& input,
            std::vector<BoxInfo>& results);
void draw_objects(const cv::Mat& image, const std::vector<yolo::Object>& objects, FILE* log_file, int frame_idx);
}



#endif /* detector_yolo_inference_hpp */
