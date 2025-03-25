#include <iostream>
#include <deque>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <chrono>
#include "detector_yolo_inference.hpp"

using namespace yolo;

// ---- CONFIG ---- //
struct Config {
    static constexpr int RAM_SIZE = 3; // 3
    static constexpr int DRM_SIZE = 6;
    static constexpr int AREA_HISTORY_SIZE = 10;
    static constexpr int UPDATE_INTERVAL = 10;
    static constexpr int DETECTION_INTERVAL = 10;
    static constexpr float IOU_THRESHOLD = 0.4;
    static constexpr float AREA_TOLERANCE = 0.9;
    static constexpr double OVERLAP_THRESHOLD = 0.4;
};

// ---- UTILS ---- //
class Utils {
public:
    static double computeIoU(const cv::Rect& r1, const cv::Rect& r2) {
        double intersection = (r1 & r2).area();
        double unionArea = r1.area() + r2.area() - intersection;
        return (intersection > 0) ? intersection / unionArea : 0.0;
    }

    static double computeMedianArea(const std::deque<double>& areas) {
        std::vector<double> sortedAreas(areas.begin(), areas.end());
        std::sort(sortedAreas.begin(), sortedAreas.end());
        size_t n = sortedAreas.size();
        return (n % 2 == 0) ? (sortedAreas[n / 2 - 1] + sortedAreas[n / 2]) / 2.0 : sortedAreas[n / 2];
    }
};

// ---- DAM MEMORY ---- //
class DAMMemory {
public:
    std::deque<cv::Rect> RAM;
    std::deque<cv::Rect> DRM;
    std::deque<double> maskAreas;

    void updateRAM(const cv::Rect& box) {
        if (RAM.size() >= Config::RAM_SIZE) RAM.pop_front();
        RAM.push_back(box);
        if (maskAreas.size() >= Config::AREA_HISTORY_SIZE) maskAreas.pop_front();
        maskAreas.push_back(box.area());
    }

    void updateDRM(const cv::Rect& box) {
        if (DRM.size() >= Config::DRM_SIZE) DRM.pop_front();
        DRM.push_back(box);
        std::cout << "Distractor saved to DRM.\n";
    }

    double getMedianArea() const {
        return Utils::computeMedianArea(maskAreas);
    }

    cv::Rect getBestMemory() const {
        if (!DRM.empty()) return DRM.back();
        if (!RAM.empty()) return RAM.back();
        return cv::Rect();
    }
};

// ---- TRACKER MANAGER ---- //
class TrackerManager {
    cv::Ptr<cv::TrackerCSRT> tracker;
public:
    TrackerManager() { tracker = cv::TrackerCSRT::create(); }

    void reinit(const cv::Mat& frame, const cv::Rect& box) {
        tracker.release();
        tracker = cv::TrackerCSRT::create();
        tracker->init(frame, box);
    }

    bool track(const cv::Mat& frame, cv::Rect& roi) {
        return tracker->update(frame, roi);
    }
};

// ---- MAIN APP ---- //
class ObjectTrackerApp {
    ncnn::Net yolov11;
    DetectorClassInfo classInfo = {1, {0}};
    DAMMemory dam;
    TrackerManager tracker;
    cv::Rect selectedROI;
    bool trackingInitialized = false;
    bool isOccluded = false;
    int frameCount = 0;
    double avg_fps = 0.0;

public:
    ObjectTrackerApp(const std::string& paramPath, const std::string& binPath) {
        yolov11.load_param(paramPath.c_str());
        yolov11.load_model(binPath.c_str());
    }

    void run(const std::string& videoPath, const std::string& outputVideoPath) {
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) return;

        int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int fps = (int)cap.get(cv::CAP_PROP_FPS);
        cv::VideoWriter writer(outputVideoPath, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), fps, cv::Size(width, height));

        cv::Mat frame;
        while (true) {
            auto t0 = std::chrono::steady_clock::now();
            cap >> frame;
            if (frame.empty()) break;
            frameCount++;

            if (frameCount == 1) initROI(frame);
            if (trackingInitialized) track(frame);
            if (frameCount % Config::DETECTION_INTERVAL == 0) detectAndUpdate(frame);

            visualize(frame);
            writer.write(frame);
            auto t1 = std::chrono::steady_clock::now();
            double frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            avg_fps = (avg_fps * (frameCount - 1) + 1000.0 / frame_time) / frameCount;
            if (cv::waitKey(1) == 'q') break;
        }
    }

private:
    void initROI(const cv::Mat& frame) {
        std::cout << "Draw ROI around target...\n";
        cv::imshow("YOLOv11 Detection", frame);
        selectedROI = cv::selectROI("YOLOv11 Detection", frame, false, false);
        cv::destroyWindow("YOLOv11 Detection");
        if (selectedROI.area() == 0) exit(0);
        tracker.reinit(frame, selectedROI);
        dam.updateRAM(selectedROI);
        trackingInitialized = true;
    }

    void track(const cv::Mat& frame) {
        if (!tracker.track(frame, selectedROI)) {
            std::cout << "CSRT lost target, marking as occluded...\n";
            trackingInitialized = false;
            isOccluded = true;
        }
    }

    void detectAndUpdate(const cv::Mat& frame) {
        std::vector<BoxInfo> detections;
        yolo::detect(classInfo, yolov11, frame, detections);
        bool targetFound = false;

        for (auto& box : detections) {
            BBox bbox = box.getBox();
            cv::Rect detectedBox(bbox.x, bbox.y, bbox.w, bbox.h);
            double iou = Utils::computeIoU(selectedROI, detectedBox);
            double medianArea = dam.getMedianArea();
            double areaDiff = std::abs(detectedBox.area() - medianArea) / medianArea;

            if (iou > Config::OVERLAP_THRESHOLD) {
                targetFound = true;
                tracker.reinit(frame, detectedBox);
                selectedROI = detectedBox;
                dam.updateRAM(detectedBox);
                trackingInitialized = true;
            } else if (iou < Config::IOU_THRESHOLD && areaDiff <= Config::AREA_TOLERANCE) {
                dam.updateDRM(detectedBox);
            }
        }

        if (!targetFound) isOccluded = true;
        else isOccluded = false;

        if (isOccluded) recover(frame);
    }

    void recover(const cv::Mat& frame) {
        cv::Rect recoveryBox = dam.getBestMemory();
        if (recoveryBox.area() > 0) {
            tracker.reinit(frame, recoveryBox);
            selectedROI = recoveryBox;
            trackingInitialized = true;
            isOccluded = false;
            std::cout << "Recovered using DAM memory.\n";
        }
    }

    void visualize(cv::Mat& frame) {
        cv::rectangle(frame, selectedROI, cv::Scalar(0, 255, 0), 2);
        std::string fps_text = "FPS: " + std::to_string(int(avg_fps));
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        cv::imshow("YOLOv11 Detection", frame);
    }
};

// ---- MAIN ---- //
int main()
{
    std::string base = "/Users/3i-a1-2022-062/workspace/yolo11_track";
    ObjectTrackerApp app(base + "/best_ncnn_model/model.ncnn.param", base + "/best_ncnn_model/model.ncnn.bin");

    std::string video = "/Users/3i-a1-2022-062/workspace/videos/occlusion-cases/2.mp4";
    std::string output = "/Users/3i-a1-2022-062/workspace/yolo11_track/output_video/output_detection.mp4";

    app.run(video, output);
    return 0;
}




//#include <iostream>
//#include <deque>
//#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
//#include <chrono>
//#include <numeric>
//#inclue "dectector_yolo_inference.hpp"
//
//using namespace yolo;
//
//// --- Parameters Configuration ---
//struct Config{
//    static constexpr int RAM_SIZE = 3;
//    static constexpr int DRM_SIZE = 6;
//    static constexpr int AREA_HISTORY_SIZE = 10;
//    static constexpr int UPDATE_INTERVAL = 10;
//    static constexpr int DETECTION_INTERVAL = 10;
//    static constexpr float IOU_THRESHOLD = 0.4;
//    static constexpr float AREA_TOLERENCE = 0.9;
//    static constexpr double OVERLAP_THRESHOLD = 0.4;
//
//};
//
//// Define Utils
//class Utils {
//public:
//    static double
//}


////// Code is working fine --20250324
//#include <iostream>
//#include "detector_yolo_inference.hpp"
//#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
//#include <chrono>
//#include <deque>
//#include <numeric>
//
//using namespace yolo;
//
//// DAM Memory
//std::deque<cv::Rect> RAM;
//std::deque<cv::Rect> DRM;
//std::deque<double> maskAreas;
//
//const int RAM_SIZE = 3; //4 default --> 3
//const int DRM_SIZE = 6; //6 default --> 3
//const int AREA_HISTORY_SIZE = 10; //10 default --> 15
//const int UPDATE_INTERVAL = 10; //5 default --> 10
//const int DETECTION_INTERVAL = 10; //10 default --> 15
//
//float IOU_THRESHOLD = 0.4; //0.8 default
//float AREA_TOLERANCE = 0.9; //0.2 default --> 0.5
//
//bool isOccluded = false;
//
//// IoU Helper
//double computeIoU(const cv::Rect& r1, const cv::Rect& r2) {
//    double intersection = (r1 & r2).area();
//    double unionArea = r1.area() + r2.area() - intersection;
//    return (intersection > 0) ? intersection / unionArea : 0.0;
//}
//
//double computeMedianArea(const std::deque<double>& areas) {
//    std::vector<double> sortedAreas(areas.begin(), areas.end());
//    std::sort(sortedAreas.begin(), sortedAreas.end());
//    size_t n = sortedAreas.size();
//    return (n % 2 == 0) ? (sortedAreas[n/2 - 1] + sortedAreas[n/2]) / 2.0 : sortedAreas[n/2];
//}
//
//int main(void)
//{
//    std::string projDir = "/Users/3i-a1-2022-062/workspace/yolo11_track";
//    std::string modelDir = projDir + "/best_ncnn_model";
//    std::string paramPath = modelDir + "/model.ncnn.param";
//    std::string binPath = modelDir + "/model.ncnn.bin";
//
//    ncnn::Net yolov11;
//    yolov11.load_param(paramPath.c_str());
//    yolov11.load_model(binPath.c_str());
//
//    DetectorClassInfo classInfo = {1, {0}};
//    
//    // video path-default: /Users/3i-a1-2022-062/workspace/videos/occlusion-cases/2.mp4
//    // 0318
//    // video path test-01: '/Users/3i-a1-2022-062/workspace/videos/testing_videos/TestVideo(2Horses) 2/2Horses_VideoCapture1.MOV'
//    // video path test-02: '/Users/3i-a1-2022-062/workspace/videos/testing_videos/TestVideo(2Horses) 2/2Horses_VideoCapture2.MOV'
//    // video path test-03: '/Users/3i-a1-2022-062/workspace/videos/testing_videos/TestVideo(2Horses) 2/2Horses_VideoCapture3(1HorseFocus&Zoom).MOV'
//    // video path test-04: '/Users/3i-a1-2022-062/workspace/videos/testing_videos/TestVideo(2Horses)/2Horses_VideoCapture5(1HorseFocus&Zoom).MOV'
//    // video path test-05: '/Users/3i-a1-2022-062/workspace/videos/testing_videos/TestVideo(2Horses)/2Horses_VideoCapture6(1HorseFocus&Zoom).MOV'
//    
//    // 0320
//    // video path test-02: '/Users/3i-a1-2022-062/workspace/videos/testing_videos/01_iPhoneMax_0.5xLens4k(72s_2horses).MOV'
//    // video path test-03: /Users/3i-a1-2022-062/workspace/videos/testing_videos/02_1_20250319.mp4 -- some time fail -- ZoomIn and ZoomOut
//    // video path test-04: /Users/3i-a1-2022-062/workspace/videos/testing_videos/02_2_20250319.mp4 -- complete fail low resolution
//    // video path test-05: /Users/3i-a1-2022-062/workspace/videos/testing_videos/02_3_20250319.mp4 --some time fail -- ZoomIn and ZoomOut
//    // video path test-06: '/Users/3i-a1-2022-062/workspace/videos/testing_videos/03_2025030502_2Horses(FHD).MOV' -- done
//    // video path test-07: /Users/3i-a1-2022-062/workspace/videos/testing_videos/06_data-2.MOV -- done
//    // video path test-07: /Users/3i-a1-2022-062/workspace/videos/testing_videos/07_IMG_3423.MOV -- done and check comparison with Insta360 and VisionAPI
//    
//    // /Users/3i-a1-2022-062/workspace/videos/occlusion-cases/2.mp4 -- check and parameters working fine
//    std::string video_path = "/Users/3i-a1-2022-062/workspace/videos/testing_videos/01_iPhoneMax_0.5xLens4k(72s_2horses).MOV";
//    cv::VideoCapture cap(video_path);
//
//    if (!cap.isOpened()) return -1;
//
//    int frame_width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
//    int frame_height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
//    int fps = (int)cap.get(cv::CAP_PROP_FPS);
//
//    std::string output_video_path = "/Users/3i-a1-2022-062/workspace/yolo11_track/output_video/output_detection.mp4";
//    cv::VideoWriter writer(output_video_path, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), fps, cv::Size(frame_width, frame_height));
//
//    int frameCount = 0;
//    cv::Mat frame;
//    cv::Ptr<cv::TrackerCSRT> tracker = cv::TrackerCSRT::create();
//    cv::Rect selectedROI;
//    bool trackingInitialized = false;
//    double avg_fps = 0.0;
//
//    while (true) {
//        auto frame_start = std::chrono::steady_clock::now();
//        cap >> frame;
//        if (frame.empty()) break;
//        frameCount++;
//
//        // First frame: manual ROI selection
//        if (frameCount == 1) {
//            std::cout << "Draw ROI around target...\n";
//            cv::imshow("YOLOv11 Detection", frame);
//            selectedROI = cv::selectROI("YOLOv11 Detection", frame, false, false);
//            cv::destroyWindow("YOLOv11 Detection");
//            if (selectedROI.area() == 0) break;
//            tracker->init(frame, selectedROI);
//            trackingInitialized = true;
//            RAM.push_back(selectedROI);
//            maskAreas.push_back(selectedROI.area());
//        }
//
//        // Step 1: CSRT continuous tracking
//        if (trackingInitialized) {
//            bool success = tracker->update(frame, selectedROI);
//            if (!success) {
//                std::cout << "CSRT lost the object, will use DAM recovery.\n";
//                trackingInitialized = false;
//                isOccluded = true;
//            }
//        }
//
//        // Step 2: periodic YOLOv11 detection + DAM update every N frames
//        if (frameCount % DETECTION_INTERVAL == 0) {
//            std::vector<BoxInfo> detections;
//            detect(classInfo, yolov11, frame, detections);
//            bool targetFound = false;
//            cv::Rect bestDetection;
//
//            for (auto& box : detections) {
//                BBox bbox = box.getBox();
//                cv::Rect detectedBox(bbox.x, bbox.y, bbox.w, bbox.h);
//                double iou = computeIoU(selectedROI, detectedBox);
//
//                if (iou > 0.4) { //0.5 default --> 0.2 (work) --> 0.4 (7video comparison) --> 0.2
//                    targetFound = true;
//                    bestDetection = detectedBox;
//                    
//                    // Update tracker with fresh detection
//                    tracker.release();
//                    tracker = cv::TrackerCSRT::create();
//                    tracker->init(frame, bestDetection);
//                    selectedROI = bestDetection;
//                    trackingInitialized = true;
//
//                    // Update RAM
//                    if (RAM.size() >= RAM_SIZE) RAM.pop_front();
//                    RAM.push_back(bestDetection);
//                    if (maskAreas.size() >= AREA_HISTORY_SIZE) maskAreas.pop_front();
//                    maskAreas.push_back(bestDetection.area());
//                }
//                else {
//                    // Distractor detected - DRM update if stable
//                    double medianArea = computeMedianArea(maskAreas);
//                    double areaDiff = std::abs(detectedBox.area() - medianArea) / medianArea;
//                    if (iou < IOU_THRESHOLD && areaDiff <= AREA_TOLERANCE) {
//                        if (DRM.size() >= DRM_SIZE) DRM.pop_front();
//                        DRM.push_back(detectedBox);
//                        std::cout << "Distractor saved to DRM.\n";
//                    }
//                }
//            }
//
//            if (!targetFound) isOccluded = true;
//            else isOccluded = false;
//        }
//
//        // Step 3: occlusion recovery from DRM or RAM
//        if (isOccluded) {
//            if (!DRM.empty()) {
//                selectedROI = DRM.back();
//                tracker = cv::TrackerCSRT::create();
//                tracker->init(frame, selectedROI);
//                trackingInitialized = true;
//                isOccluded = false;
//                std::cout << "Recovered using DRM!\n";
//            } else if (!RAM.empty()) {
//                selectedROI = RAM.back();
//                tracker = cv::TrackerCSRT::create();
//                tracker->init(frame, selectedROI);
//                trackingInitialized = true;
//                isOccluded = false;
//                std::cout << "Fallback recovery using RAM.\n";
//            }
//        }
//
//        // Step 4: Visualization
//        cv::rectangle(frame, selectedROI, cv::Scalar(0, 255, 0), 2);
//
//        auto frame_end = std::chrono::steady_clock::now();
//        double frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
//        double fps_calculated = 1000.0 / frame_time;
//        avg_fps = (avg_fps * (frameCount - 1) + fps_calculated) / frameCount;
//
//        std::string fps_text = "FPS: " + std::to_string(int(avg_fps));
//        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
//
//        cv::imshow("YOLOv11 Detection", frame);
//        writer.write(frame);
//
//        if (cv::waitKey(1) == 'q') break;
//    }
//
//    cap.release();
//    writer.release();
//    cv::destroyAllWindows();
//    return 0;
//}



 //--code is working fine also handling occlusion cases-- 20250319
//#include <iostream>
//#include "detector_yolo_inference.hpp"
//#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
//#include <chrono>
//#include <deque>
//#include <numeric>
//
//using namespace yolo;
//
//// DAM Memory
//std::deque<cv::Rect> RAM;
//std::deque<cv::Rect> DRM;
//std::deque<double> maskAreas;
//
//const int RAM_SIZE = 4;
//const int DRM_SIZE = 6;
//const int AREA_HISTORY_SIZE = 10;
//const int UPDATE_INTERVAL = 5;
//const int DETECTION_INTERVAL = 10;
//
//float IOU_THRESHOLD = 0.8;
//float AREA_TOLERANCE = 0.2;
//
//bool isOccluded = false;
//
//// IoU Helper
//double computeIoU(const cv::Rect& r1, const cv::Rect& r2) {
//    double intersection = (r1 & r2).area();
//    double unionArea = r1.area() + r2.area() - intersection;
//    return (intersection > 0) ? intersection / unionArea : 0.0;
//}
//
//double computeMedianArea(const std::deque<double>& areas) {
//    std::vector<double> sortedAreas(areas.begin(), areas.end());
//    std::sort(sortedAreas.begin(), sortedAreas.end());
//    size_t n = sortedAreas.size();
//    return (n % 2 == 0) ? (sortedAreas[n/2 - 1] + sortedAreas[n/2]) / 2.0 : sortedAreas[n/2];
//}
//
//int main(void)
//{
//    std::string projDir = "/Users/3i-a1-2022-062/workspace/yolo11_track";
//    std::string modelDir = projDir + "/best_ncnn_model";
//    std::string paramPath = modelDir + "/model.ncnn.param";
//    std::string binPath = modelDir + "/model.ncnn.bin";
//
//    ncnn::Net yolov11;
//    yolov11.load_param(paramPath.c_str());
//    yolov11.load_model(binPath.c_str());
//
//    DetectorClassInfo classInfo = {1, {0}};
//    std::string video_path = "/Users/3i-a1-2022-062/workspace/videos/occlusion-cases/2.mp4";
//    cv::VideoCapture cap(video_path);
//
//    if (!cap.isOpened()) return -1;
//
//    int frame_width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
//    int frame_height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
//    int fps = (int)cap.get(cv::CAP_PROP_FPS);
//
//    std::string output_video_path = "/Users/3i-a1-2022-062/workspace/yolo11_track/output_video/output_detection.mp4";
//    cv::VideoWriter writer(output_video_path, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), fps, cv::Size(frame_width, frame_height));
//
//    int frameCount = 0;
//    cv::Mat frame;
//    cv::Ptr<cv::TrackerCSRT> tracker = cv::TrackerCSRT::create();
//    cv::Rect selectedROI;
//    bool trackingInitialized = false;
//    double avg_fps = 0.0;
//
//    while (true) {
//        auto frame_start = std::chrono::steady_clock::now();
//        cap >> frame;
//        if (frame.empty()) break;
//        frameCount++;
//
//        // First frame: manual ROI selection
//        if (frameCount == 1) {
//            std::cout << "Draw ROI around target...\n";
//            cv::imshow("YOLOv11 Detection", frame);
//            selectedROI = cv::selectROI("YOLOv11 Detection", frame, false, false);
//            cv::destroyWindow("YOLOv11 Detection");
//            if (selectedROI.area() == 0) break;
//            tracker->init(frame, selectedROI);
//            trackingInitialized = true;
//            RAM.push_back(selectedROI);
//            maskAreas.push_back(selectedROI.area());
//        }
//
//        // Step 1: CSRT continuous tracking
//        if (trackingInitialized) {
//            bool success = tracker->update(frame, selectedROI);
//            if (!success) {
//                std::cout << "CSRT lost the object, will use DAM recovery.\n";
//                trackingInitialized = false;
//                isOccluded = true;
//            }
//        }
//
//        // Step 2: periodic YOLOv11 detection + DAM update every N frames
//        if (frameCount % DETECTION_INTERVAL == 0) {
//            std::vector<BoxInfo> detections;
//            detect(classInfo, yolov11, frame, detections);
//            bool targetFound = false;
//            cv::Rect bestDetection;
//
//            for (auto& box : detections) {
//                BBox bbox = box.getBox();
//                cv::Rect detectedBox(bbox.x, bbox.y, bbox.w, bbox.h);
//                double iou = computeIoU(selectedROI, detectedBox);
//
//                if (iou > 0.5) {
//                    targetFound = true;
//                    bestDetection = detectedBox;
//                    
//                    // Update tracker with fresh detection
//                    tracker.release();
//                    tracker = cv::TrackerCSRT::create();
//                    tracker->init(frame, bestDetection);
//                    selectedROI = bestDetection;
//                    trackingInitialized = true;
//
//                    // Update RAM
//                    if (RAM.size() >= RAM_SIZE) RAM.pop_front();
//                    RAM.push_back(bestDetection);
//                    if (maskAreas.size() >= AREA_HISTORY_SIZE) maskAreas.pop_front();
//                    maskAreas.push_back(bestDetection.area());
//                }
//                else {
//                    // Distractor detected - DRM update if stable
//                    double medianArea = computeMedianArea(maskAreas);
//                    double areaDiff = std::abs(detectedBox.area() - medianArea) / medianArea;
//                    if (iou < IOU_THRESHOLD && areaDiff <= AREA_TOLERANCE) {
//                        if (DRM.size() >= DRM_SIZE) DRM.pop_front();
//                        DRM.push_back(detectedBox);
//                        std::cout << "Distractor saved to DRM.\n";
//                    }
//                }
//            }
//
//            if (!targetFound) isOccluded = true;
//            else isOccluded = false;
//        }
//
//        // Step 3: occlusion recovery from DRM or RAM
//        if (isOccluded) {
//            if (!DRM.empty()) {
//                selectedROI = DRM.back();
//                tracker = cv::TrackerCSRT::create();
//                tracker->init(frame, selectedROI);
//                trackingInitialized = true;
//                isOccluded = false;
//                std::cout << "Recovered using DRM!\n";
//            } else if (!RAM.empty()) {
//                selectedROI = RAM.back();
//                tracker = cv::TrackerCSRT::create();
//                tracker->init(frame, selectedROI);
//                trackingInitialized = true;
//                isOccluded = false;
//                std::cout << "Fallback recovery using RAM.\n";
//            }
//        }
//
//        // Step 4: Visualization
//        cv::rectangle(frame, selectedROI, cv::Scalar(0, 255, 0), 2);
//
//        auto frame_end = std::chrono::steady_clock::now();
//        double frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
//        double fps_calculated = 1000.0 / frame_time;
//        avg_fps = (avg_fps * (frameCount - 1) + fps_calculated) / frameCount;
//
//        std::string fps_text = "FPS: " + std::to_string(int(avg_fps));
//        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
//
//        cv::imshow("YOLOv11 Detection", frame);
//        writer.write(frame);
//
//        if (cv::waitKey(1) == 'q') break;
//    }
//
//    cap.release();
//    writer.release();
//    cv::destroyAllWindows();
//    return 0;
//}
