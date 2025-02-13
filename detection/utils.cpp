//
//  utils.cpp
//  Inference
//
//  Created by Song on 2/5/24.
//

#include "utils.h"

#include <fstream>
#include <sstream>

const int color_list[3][3] = {
    //{255 ,255 ,255}, //bg
    {255, 0, 0}, // Blue
    {200, 255, 100}, // Green
    {0, 0, 255}}; // Red


namespace dateTime {
    std::string currentDateTime() {
        time_t now = time(0);
        struct tm tstruct;
        char buf[80];
        tstruct = *localtime(&now);
        strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
        return buf;
    }

}

namespace filePath {
std::string
getStem(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    std::string filename = lastSlash != std::string::npos ? path.substr(lastSlash + 1) : path;
    
    size_t dotPos = filename.find_last_of('.');
    if (dotPos != std::string::npos) {
        return filename.substr(0, dotPos);
    }
    
    return filename;
}

std::string
getParentDirectory(const std::string& filename) {
    size_t pos = filename.find_last_of("/\\");
    if (pos != std::string::npos) {
        return filename.substr(0, pos);
    } else {
        return "";
    }
}

std::string getLowerCase(std::string const& filename) {
    std::string lowercase_name = filename;
    std::transform(lowercase_name.begin(), lowercase_name.end(), lowercase_name.begin(), [](unsigned char c) {return std::tolower(c); });
    
    return lowercase_name;
}

std::string
getFileExtension (std::string const& filename)
{
    std::size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos)
        return std::string();
    
    return getLowerCase(filename.substr(pos));
}
}

namespace draw {
void drawBoundingBox(const cv::Mat &image, std::vector<BoxInfo> &bboxes, const float& fps, const std::size_t& count, const std::size_t& missCount, const std::size_t& twoTargetMissCount, const std::size_t& numSelectedDetBox, const std::size_t numTrackedBox, const bool& do_group )
{
    static const std::array<std::string, 3> class_names =  {"person",  "tennis racket"};
    
    const auto src_w = image.cols;
    const auto src_h = image.rows;
    auto baseline = int{10};
    std::string text;
    cv::Size label_size = {10, 10};
    for (BoxInfo &bbox : bboxes)
    {
        auto score = bbox.getConfidence();
        // auto trackId = bbox.getId();
        auto classId = bbox.getClassId();
        auto boxData = bbox.getBox();
        auto rectData = cv::Rect(boxData.x, boxData.y, boxData.w, boxData.h);
        auto xc = bbox.getCenter().xc;
        auto yc = bbox.getCenter().yc;
        
        const auto color = cv::Scalar(color_list[0][classId]);
        
        cv::rectangle(image, rectData, color);
        
        auto x = boxData.x;
        auto y = boxData.y - 10;
        
        text = "[" + std::to_string(classId) + "] "
        + std::to_string(int(score *100) % 100) + "% "
        + " W" + std::to_string(rectData.width) + ", H" + std::to_string(rectData.height) + " "
        + " (" + std::to_string(boxData.x) + ", " + std::to_string(boxData.y) + ") ";
        
        if (bbox.getConfidence() * 100 < 1 && do_group)
            text = " Grouped ";
        
        label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseline);
        
        
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;
        if (bbox.getConfidence() * 100 < 1 && do_group) {
            
            
            cv::rectangle(image, rectData, cv::Scalar(0,0,0), 2);
            cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                          cv::Size(label_size.width, 2* label_size.height + baseline)),
                          cv::Scalar(0, 0, 0), -1);
            
            cv::putText(image, text, cv::Point(x, y + 2* label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        }
        else {
            cv::rectangle(image, rectData, color);
            cv::rectangle(
                          image,
                          cv::Rect(cv::Point(x, y),
                                   cv::Size(label_size.width, label_size.height + baseline)),
                          color, -1);
            cv::putText(image, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
        }
        
    }
    cv::putText(image, "fps: " + std::to_string(int(fps)), cv::Point(50, 50 + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 3);
    cv::putText(image, "fps: " + std::to_string(int(fps)), cv::Point(50, 50 + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(image, "fi: " + std::to_string(count), cv::Point(src_w - 150,  50 + label_size.height ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 3);
    cv::putText(image, "fi: " + std::to_string(count), cv::Point(src_w - 150,  50 + label_size.height ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    cv::putText(image, "miss1: " + std::to_string(missCount), cv::Point(50,  src_h - (50 + label_size.height) ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 3);
    cv::putText(image, "miss1: " + std::to_string(missCount), cv::Point(50, src_h - (50 + label_size.height) ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    cv::putText(image, "miss2: " + std::to_string(twoTargetMissCount), cv::Point(50,  src_h - (150 + label_size.height) ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 3);
    cv::putText(image, "miss2: " + std::to_string(twoTargetMissCount), cv::Point(50, src_h - (150 + label_size.height) ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    
    cv::putText(image, "n(selDet): " + std::to_string(numSelectedDetBox), cv::Point(src_w - 200, src_h - (50 + label_size.height) ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 3);
    cv::putText(image, "n(selDet): " + std::to_string(numSelectedDetBox), cv::Point(src_w - 200, src_h - (50 + label_size.height) ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    cv::putText(image, "n(track): " + std::to_string(numTrackedBox), cv::Point(src_w - 200, src_h - (150 + label_size.height) ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 3);
    cv::putText(image, "n(track): " + std::to_string(numTrackedBox), cv::Point(src_w - 200, src_h - (150 + label_size.height) ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
}

}

namespace bboxUtils {


void sortBoundingBox(const std::size_t maximum_number, BoxInfoVector& detectedBBoxes) {
    std::sort(detectedBBoxes.begin(), detectedBBoxes.end(), [](BoxInfo& a, BoxInfo& b) {
        return a.getConfidence() > b.getConfidence();
    });
    
    if (detectedBBoxes.size() > maximum_number) {
        std::sort(detectedBBoxes.begin() + 1, detectedBBoxes.end(),[](BoxInfo& a, BoxInfo& b) {
            return (a.getBox().w * a.getBox().h) > (b.getBox().w * b.getBox().h);
        });
    }
}


double calculateIoU(BoxInfo &obj0, BoxInfo &obj1) {
    int32_t interx0 = (std::max)(obj0.getLeft(), obj1.getLeft());
    int32_t intery0 = (std::max)(obj0.getTop(), obj1.getTop());
    int32_t interx1 = (std::min)(obj0.getRight(), obj1.getRight());
    int32_t intery1 = (std::min)(obj0.getBottom(), obj1.getBottom());
    if (interx1 < interx0 || intery1 < intery0) return 0;

    int32_t area0 = (obj0.getWidth()) * (obj0.getHeight());
    int32_t area1 = (obj1.getWidth()) * (obj1.getHeight());
    int32_t areaInter = (interx1 - interx0) * (intery1 - intery0);
    int32_t areaSum = area0 + area1 - areaInter;

    return static_cast<double>(areaInter) / areaSum;
}

double calculateOverlap(BoxInfo target, BoxInfo other) {
    int32_t interx0 = (std::max)(target.getLeft(), other.getLeft());
    int32_t intery0 = (std::max)(target.getTop(), other.getTop());
    int32_t interx1 = (std::min)(target.getRight(), other.getRight());
    int32_t intery1 = (std::min)(target.getBottom(), other.getBottom());
    if (interx1 < interx0 || intery1 < intery0) return 0;

    int32_t areaTarget = (target.getWidth()) * (target.getHeight());
    int32_t areaOther = (other.getWidth()) * (other.getHeight());
    int32_t areaInter = (interx1 - interx0) * (intery1 - intery0);

    return static_cast<double>(areaInter) / areaTarget;
}

double calculateOverlap(BBox target, BBox other) {
    int32_t interx0 = (std::max)(target.x, other.x);
    int32_t intery0 = (std::max)(target.y, other.y);
    int32_t interx1 = (std::min)(target.x + target.w, other.x + other.w);
    int32_t intery1 = (std::min)(target.y + target.h, other.y + other.h);
    if (interx1 < interx0 || intery1 < intery0) return 0;

    int32_t areaTarget = (target.w) * (target.h);
    int32_t areaOther = (other.w) * (other.h);
    int32_t areaInter = (interx1 - interx0) * (intery1 - intery0);

    return static_cast<double>(areaInter) / areaTarget;
}


double calculateOverlap(cv::Rect target, BBox other) {
    if (target.empty())
        return 0;
    
    int32_t interx0 = (std::max)(target.x, other.x);
    int32_t intery0 = (std::max)(target.y, other.y);
    int32_t interx1 = (std::min)(target.x + target.width, other.x + other.w);
    int32_t intery1 = (std::min)(target.y + target.height, other.y + other.h);
    if (interx1 < interx0 || intery1 < intery0) return 0;

    int32_t areaTarget = (target.width) * (target.height);
    int32_t areaOther = (other.w) * (other.h);
    int32_t areaInter = (interx1 - interx0) * (intery1 - intery0);

    return static_cast<double>(areaInter) / areaTarget;
}

double computeDistance(const BBoxCenter& center1, const BBoxCenter& center2) {
    return sqrt(pow(center2.xc - center1.xc, 2) + pow(center2.yc - center1.yc, 2));
}


cv::Point getRectCenter(const cv::Rect& input) {
    return {static_cast<int>(input.x + input.width/2.0), static_cast<int>(input.y + input.height/2.0)};
}

bool isPointInside(const cv::Rect& inputBox, const cv::Point& inputPoint) {
    if (inputPoint.x >= inputBox.x &&
        inputPoint.y >= inputBox.y &&
        inputPoint.x <= (inputBox.x + inputBox.width) &&
        inputPoint.y <= (inputBox.y + inputBox.height) )
        return true;
    
    return false;
}


int getConfidenceScoreRank(double score) {
    std::vector<std::pair<double, int>> boundaries = {
        {0.50, 10},
        {0.55, 9},
        {0.60, 8},
        {0.65, 7},
        {0.70, 6},
        {0.75, 5},
        {0.80, 4},
        {0.85, 3},
        {0.90, 2},
        {0.95, 1},
        {1.00, 0},
    };

    for (const auto& boundary : boundaries) {
        if (score <= boundary.first)
            return boundary.second;
    }

    // Return a default rank if score doesn't fit any range
    return 11; // Larger number than current rank.
}

int getDistanceRank(double distance, double thresh) {
    double normalizedDist = distance / thresh;
    
    std::vector<std::pair<double, int>> boundaries = {
        {0.01, 0},
        {0.02, 1},
        {0.04, 2},
        {0.07, 3},
        {0.12, 4},
        {0.19, 5},
        {0.27, 6},
        {0.35, 7},
        {0.44, 8},
        {0.54, 9},
        {0.65, 10},
        {0.77, 11},
        {1.00, 12}
    };

    for (const auto& boundary : boundaries) {
        if (normalizedDist <= boundary.first)
            return boundary.second;
    }

    // Return a default rank if score doesn't fit any range
    return 15; // Indicates an error or not found by case
}

int getDistanceFromRank(double distance) {
    double normalizedDist = distance;
    
    std::vector<std::pair<double, int>> boundaries = {
        {0.05, 0},
        {0.1, 1},
        {0.2, 2},
        {0.3, 3},
        {0.4, 4},
        {0.5, 5},
        {0.55, 6},
        {0.60, 7},
        {0.65, 8},
        {0.70, 9},
        {0.75, 10},
        {0.85, 11},
        {0.95, 12}
    };

    for (const auto& boundary : boundaries) {
        if (normalizedDist <= boundary.first)
            return boundary.second;
    }

    // Return a default rank if score doesn't fit any range
    return 15; // Indicates an error or not found by case
}
int getAreaRank(double area, double maxArea) {
    double normalizedArea = area / maxArea;
    
    std::vector<std::pair<double, int>> boundaries = {
        {0.0, 10},
        {0.1, 9},
        {0.4, 6},
        {0.7, 3},
        {1.0, 0}
    };
    
    for (const auto& boundary : boundaries) {
        if (normalizedArea <= boundary.first)
            return boundary.second;
    }

    // Return a default rank if score doesn't fit any range
    return 11; // Indicates an error or not found
}

int getOverlapRank(cv::Rect prevBox, BBox currBBox) {
    if (prevBox.empty())
        return 0;
    double overlap = calculateOverlap(prevBox, currBBox);
    std::vector<std::pair<double, int>> boundaries = {
        {0.01, 10},
        {0.1, 9},
        {0.2, 8},
        {0.3, 7},
        {0.4, 6},
        {0.5, 5},
        {0.6, 4},
        {0.7, 3},
        {0.8, 2},
        {0.9, 1},
        {1.01, 0}
    };
    
    for (const auto& boundary : boundaries) {
        if (overlap <= boundary.first)
            return boundary.second;
    }

    return 11;

    
}

void setIntersectionBox(BoxInfo boxInfo1, BoxInfo boxInfo2, cv::Rect& box) {
    
    int xmin = std::max(boxInfo1.getLeft(), boxInfo2.getLeft());
    int ymin = std::max(boxInfo1.getTop() , boxInfo2.getTop());
    int xmax = std::min(boxInfo1.getRight(), boxInfo2.getRight());
    int ymax = std::min(boxInfo1.getBottom(), boxInfo2.getBottom());
    
    box = {xmin, ymin, xmax - xmin, ymax - ymin};
}
}