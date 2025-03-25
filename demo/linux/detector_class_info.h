//
//  detector_class_info.h
//  softzoom
//
//  Created by Song on 1/20/25.
//

#ifndef detector_class_info_h
#define detector_class_info_h

#include <iostream>
#include <unordered_set>

struct DetectorClassInfo {
    int numClasses;
    std::unordered_set<int> targetLabels;
};

#endif /* detector_class_info_h */

