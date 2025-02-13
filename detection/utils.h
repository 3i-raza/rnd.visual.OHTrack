//
//  utils.h
//  MAC_PIVO_OD
//
//  Created by Songeun Kim on 2/2/24.
//

#ifndef utils_h
#define utils_h

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "boxinfo.h"


typedef std::vector<BoxInfo> BoxInfoVector;


namespace filePath {
std::string
getStem(const std::string& path);

std::string
getParentDirectory(const std::string& filename);

std::string 
getLowerCase(std::string const& filename);

std::string
getFileExtension (std::string const& filename);
}

namespace dateTime {
std::string currentDateTime();
}

namespace draw {
void drawBoundingBox(const cv::Mat &image, std::vector<BoxInfo> &bboxes, const float& fps, const std::size_t& count, const std::size_t& missCount, const std::size_t& twoTargetMissCount, const std::size_t& numSelectedDetBox, const std::size_t numTrackedBox, const bool& do_group );
}


namespace bboxUtils {

void sortBoundingBox(const std::size_t maximum_number, BoxInfoVector& detectedBBoxes);
//void selectTargetBox(const std::size_t& targetSize, BoxInfoVector& sortedBoxes,  BoxInfoVector& selected);


template <typename T>
void convertToRect(const std::vector<BoxInfo>& boxes, std::vector<T>& rds) {
    for (BoxInfo boxInfo: boxes) {
        BBox box = boxInfo.getBox();
        rds.push_back(T(box.x, box.y, box.w, box.h));
    }
}


template <typename T>
void convertToRect(const BoxInfo& boxInfo, T& rd) {
    BBox box = boxInfo.getBox();
    rd = T(box.x, box.y, box.w, box.h);
}
template <typename T>
void clearBox(T& rd) {
    rd = T({0, 0, 0, 0});
}

template <typename T>
BBox convertToBBox(const T& rd) {
    return {rd.x, rd.y, rd.width, rd.height};
}

double calculateIoU(BoxInfo &obj0, BoxInfo &obj1);
double calculateOverlap(BoxInfo target, BoxInfo other);
double calculateOverlap(cv::Rect target, BBox other);
double computeDistance(const BBoxCenter& center1, const BBoxCenter& center2);

cv::Point getRectCenter(const cv::Rect& input);
bool isPointInside(const cv::Rect& inputBox, const cv::Point& inputPoint);

int getConfidenceScoreRank(double score);
int getDistanceRank(double distance, double thresh);
int getDistanceFromRank(double distance);
int getAreaRank(double area, double maxArea);
int getOverlapRank(cv::Rect prevBox, BBox currBBox);
void setIntersectionBox(BoxInfo boxInfo1, BoxInfo boxInfo2, cv::Rect& box);
 }


#endif /* utils_h */