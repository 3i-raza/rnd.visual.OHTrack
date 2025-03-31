// run-time RoI code
#include <iostream>
#include <deque>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <chrono>
#include "detector_yolo_inference.hpp"

using namespace yolo;

// ------------------- CONFIG --------------------- //
struct Config {
    static constexpr int RAM_SIZE = 3;
    static constexpr int DRM_SIZE = 6;
    static constexpr int AREA_HISTORY_SIZE = 10;
    static constexpr int DETECTION_INTERVAL = 10;
    static constexpr float IOU_THRESHOLD = 0.4;
    static constexpr float AREA_TOLERANCE = 0.9;
    static constexpr float OVERLAP_THRESHOLD = 0.4;
};

// ------------------- UTILS --------------------- //
class Utils {
public:
    static double computeIoU(const cv::Rect& r1, const cv::Rect& r2) {
        double intersection = (r1 & r2).area();
        double unionArea = r1.area() + r2.area() - intersection;
        return (intersection > 0) ? intersection / unionArea : 0.0;
    }

    static double computeMedianArea(const std::deque<double>& areas) {
        std::vector<double> sorted(areas.begin(), areas.end());
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        return (n % 2 == 0) ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 : sorted[n / 2];
    }
};

// ------------------- DAM MEMORY --------------------- //
class DAMMemory {
public:
    std::deque<cv::Rect> RAM, DRM;
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

// ------------------- TRACKER --------------------- //
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

// ------------------- MAIN APP --------------------- //
class ObjectTrackerApp {
    ncnn::Net yolov11;
    DetectorClassInfo classInfo = {1, {0}};
    DAMMemory dam;
    TrackerManager tracker;
    cv::Rect selectedROI;
    bool roiReady = false;
    bool trackingInitialized = false;
    bool isOccluded = false;
    int frameCount = 0;
    double avg_fps = 0.0;
    std::string windowName = "YOLOv11 Horse Tracking";

    // Mouse callback vars
    cv::Point roiStart, roiEnd;
    bool drawing = false;

public:
    ObjectTrackerApp(const std::string& paramPath, const std::string& binPath) {
        yolov11.load_param(paramPath.c_str());
        yolov11.load_model(binPath.c_str());
    }

    void run(const std::string& outputPath) {
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open webcam.\n";
            return;
        }

        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
        if (fps == 0) fps = 30;

        cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
        cv::namedWindow(windowName);
        cv::setMouseCallback(windowName, mouseCallback, this);

        cv::Mat frame;

        while (true) {
            auto t0 = std::chrono::steady_clock::now();
            cap >> frame;
            if (frame.empty()) break;
            frameCount++;

            // Visual drawing feedback
            if (drawing) {
                cv::rectangle(frame, roiStart, roiEnd, cv::Scalar(255, 255, 0), 2);
            }

            if (roiReady) validateAndInitROI(frame);
            if (trackingInitialized) {
                track(frame);
                if (frameCount % Config::DETECTION_INTERVAL == 0)
                    detectAndUpdate(frame);
            }

            visualize(frame);
            writer.write(frame);

            auto t1 = std::chrono::steady_clock::now();
            double frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            avg_fps = (avg_fps * (frameCount - 1) + 1000.0 / frame_time) / frameCount;

            if (cv::waitKey(1) == 'q') break;
        }
    }

private:
    static void mouseCallback(int event, int x, int y, int flags, void* userdata) {
        ObjectTrackerApp* app = reinterpret_cast<ObjectTrackerApp*>(userdata);
        if (event == cv::EVENT_LBUTTONDOWN) {
            app->drawing = true;
            app->roiStart = cv::Point(x, y);
            app->roiEnd = cv::Point(x, y);
        } else if (event == cv::EVENT_MOUSEMOVE && app->drawing) {
            app->roiEnd = cv::Point(x, y);
        } else if (event == cv::EVENT_LBUTTONUP && app->drawing) {
            app->drawing = false;
            app->selectedROI = cv::Rect(app->roiStart, app->roiEnd);
            app->roiReady = true;
        }
    }

    void validateAndInitROI(const cv::Mat& frame) {
        roiReady = false;
        if (selectedROI.area() <= 0) {
            std::cout << "⚠️ Empty ROI ignored.\n";
            return;
        }

        std::vector<BoxInfo> detections;
        yolo::detect(classInfo, yolov11, frame, detections);

        for (auto& box : detections) {
            BBox bbox = box.getBox();
            cv::Rect detectedBox(bbox.x, bbox.y, bbox.w, bbox.h);
            double iou = Utils::computeIoU(selectedROI, detectedBox);
            if (iou > 0.4) {
                tracker.reinit(frame, detectedBox);
                selectedROI = detectedBox;
                dam.updateRAM(detectedBox);
                trackingInitialized = true;
                isOccluded = false;
                std::cout << "✅ Horse ROI validated and tracking started.\n";
                return;
            }
        }

        std::cout << "❌ No horse found in selected region. Try again.\n";
    }

    void track(const cv::Mat& frame) {
        if (!tracker.track(frame, selectedROI)) {
            std::cout << "❌ Tracker lost target. Attempting recovery...\n";
            trackingInitialized = false;
            isOccluded = true;
        }
    }

    void detectAndUpdate(const cv::Mat& frame) {
        std::vector<BoxInfo> detections;
        yolo::detect(classInfo, yolov11, frame, detections);
        bool found = false;

        for (auto& box : detections) {
            BBox bbox = box.getBox();
            cv::Rect detBox(bbox.x, bbox.y, bbox.w, bbox.h);
            double iou = Utils::computeIoU(selectedROI, detBox);
            double areaDiff = std::abs(detBox.area() - dam.getMedianArea()) / dam.getMedianArea();

            if (iou > Config::OVERLAP_THRESHOLD) {
                tracker.reinit(frame, detBox);
                selectedROI = detBox;
                dam.updateRAM(detBox);
                trackingInitialized = true;
                found = true;
                break;
            } else if (iou < Config::IOU_THRESHOLD && areaDiff <= Config::AREA_TOLERANCE) {
                dam.updateDRM(detBox);
            }
        }

        isOccluded = !found;
        if (isOccluded) recover(frame);
    }

    void recover(const cv::Mat& frame) {
        cv::Rect memoryBox = dam.getBestMemory();
        if (memoryBox.area() > 0) {
            tracker.reinit(frame, memoryBox);
            selectedROI = memoryBox;
            trackingInitialized = true;
            isOccluded = false;
            std::cout << "✅ Recovered from DAM memory.\n";
        }
    }

    void visualize(cv::Mat& frame) {
        if (trackingInitialized)
            cv::rectangle(frame, selectedROI, cv::Scalar(0, 255, 0), 2);

        std::string fps_text = "FPS: " + std::to_string(int(avg_fps));
        std::string info_text = trackingInitialized ? "Tracking: horse (Press 'q' to quit)" : "Draw box to track horse";
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, info_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

        cv::imshow(windowName, frame);
    }
};

// ------------------- MAIN --------------------- //
int main() {
    std::string base = "/Users/3i-a1-2022-062/workspace/yolo11_track";
    ObjectTrackerApp app(
        base + "/best_ncnn_model/model.ncnn.param",
        base + "/best_ncnn_model/model.ncnn.bin"
    );

    std::string output = base + "/output_video/live_horse_tracking.avi";
    app.run(output);
    return 0;
}
