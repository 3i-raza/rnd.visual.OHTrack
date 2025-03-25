//
//  detector_yolo_inference.cpp
//  Inference
//
//  Created by Song on 12/5/24.
//

#include "detector_yolo_inference.hpp"

namespace yolo {
float intersection_area(const yolo::Object& a, const yolo::Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void qsort_descent_inplace(std::vector<yolo::Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;
    
    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;
        
        while (objects[j].prob < p)
            j--;
        
        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);
            
            i++;
            j--;
        }
    }
    
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<yolo::Object>& objects)
{
    if (objects.empty())
        return;
    
    qsort_descent_inplace(objects, 0, int(objects.size() - 1));
}

void nms_sorted_bboxes(const std::vector<yolo::Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic)
{
    picked.clear();
    
    const int n = int(faceobjects.size());
    
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }
    
    for (int i = 0; i < n; i++)
    {
        const yolo::Object& a = faceobjects[i];
        
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const yolo::Object& b = faceobjects[picked[j]];
            
            if (!agnostic && a.label != b.label)
                continue;
            
            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float IoU = inter_area / union_area;
            
            if (inter_area / union_area > nms_threshold) {
                keep = 0;
            }

        }
        
        if (keep) {
            picked.push_back(i);
        }
    }
}

float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

float clampf(float d, float min, float max)
{
    const float t = d < min ? min : d;
    return t > max ? max : t;
}

void parse_yolov_detections(
                                   float* inputs, float confidence_threshold,
                                   int num_channels, int num_anchors, int num_labels,
                                   int infer_img_width, int infer_img_height,
                                   std::vector<yolo::Object>& objects)
{
    std::vector<yolo::Object> detections;
    
    cv::Mat output = cv::Mat((int)num_channels, (int)num_anchors, CV_32F, inputs).t();
    
    for (int i = 0; i < num_anchors; i++)
    {
        const float* row_ptr = output.row(i).ptr<float>();
        const float* bboxes_ptr = row_ptr; // 0x140e2b09c    0x0000000140e2b0 9c
        const float* scores_ptr = row_ptr + 4; // 0x140e2b0a0    0x0000000140e2b0 a0
        const float* max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_labels);
        float score = *max_s_ptr;
        
        if (score > confidence_threshold)
        {
            float x = *bboxes_ptr++;
            float y = *bboxes_ptr++;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;
            
            float x0 = clampf((x - 0.5f * w), 0.f, (float)infer_img_width);
            float y0 = clampf((y - 0.5f * h), 0.f, (float)infer_img_height);
            float x1 = clampf((x + 0.5f * w), 0.f, (float)infer_img_width);
            float y1 = clampf((y + 0.5f * h), 0.f, (float)infer_img_height);
            
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;
            yolo::Object object;
            object.label = int(max_s_ptr - scores_ptr);
            object.prob = score;
            object.rect = bbox;
            detections.push_back(object);
        }
    }
    objects = detections;
}


void detect(const DetectorClassInfo classInfo, ncnn::Net& yoloModel, const cv::Mat& input, std::vector<BoxInfo>& results) {
    const int target_size = 640;
    const float prob_threshold = PROB_THRESHOLD;
    const float nms_threshold = 0.45f;
    
    int img_w = input.cols;
    int img_h = input.rows;
    
    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(input.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
    int wpad = (target_size + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (target_size + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);
    
    ncnn::Extractor ex = yoloModel.create_extractor();
    
    
    // Input 3 in_pad's result
    ex.input("in0", in_pad);
    
    std::vector<yolo::Object> objects;
    std::vector<yolo::Object> proposals;
    
    // stride 32
    {
        ncnn::Mat out;
        ex.extract("out0", out, 0);
        
        std::vector<yolo::Object> objects32;
        parse_yolov_detections(
                               (float*)out.data, float(prob_threshold),
                               int(out.h), int(out.w), int(classInfo.numClasses),
                               int(in_pad.w), int(in_pad.h),
                               objects32);
        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }
    
    qsort_descent_inplace(proposals);
    
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    
    objects.resize(picked.size());
    
    for (int i = 0; i < picked.size(); i++)
    {
        if (classInfo.targetLabels.find(proposals[picked[i]].label) != classInfo.targetLabels.end()) {
            objects[i] = proposals[picked[i]];
            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
            float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
            
            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
            
            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
            
            results.push_back(BoxInfo(-1, objects[i].label, objects[i].prob, BBox(x0, y0, x1-x0, y1-y0)));
        }
    }
}



void draw_objects(const cv::Mat& image, const std::vector<Object>& objects, FILE* log_file, int frame_idx)
{
    
    static const unsigned char colors[19][3] = {
        {54, 67, 244},
        {99, 30, 233},
        {176, 39, 156},
        {183, 58, 103},
        {181, 81, 63},
        {243, 150, 33},
        {244, 169, 3},
        {212, 188, 0},
        {136, 150, 0},
        {80, 175, 76},
        {74, 195, 139},
        {57, 220, 205},
        {59, 235, 255},
        {7, 193, 255},
        {0, 152, 255},
        {34, 87, 255},
        {72, 85, 121},
        {158, 158, 158},
        {139, 125, 96}
    };
    
    int color_index = 0;
    
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        
        const unsigned char* color = colors[color_index % 19];
        color_index++;
        
        cv::Scalar cc(color[0], color[1], color[2]);
        
        
        cv::rectangle(image, obj.rect, cc, 2);
        
        char text[256];
        sprintf(text, "%s %.1f%% (%.0f, %.0f) ", class_names[obj.label], obj.prob * 100, obj.rect.x, obj.rect.y);
        
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;
        
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cc, -1);
        
        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        
    }
    cv::imshow("image", image);
    std::string out_file_path = "/Users/3i-21-331/workspace/New Folder With Items 5/Result.png";
    cv::imwrite(out_file_path, image);
    cv::waitKey(0);
}
}
