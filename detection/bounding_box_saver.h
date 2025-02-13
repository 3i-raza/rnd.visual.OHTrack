#include <nlohmann/json.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;

class BoundingBoxSaver {
public:
    static void saveBoundingBox(const std::string& filename, const cv::Rect& bbox, 
                              const std::string& image_path) {
        try {
            // Create JSON object
            json j;
            
            // Add image information
            j["image_path"] = image_path;
            
            // Add bounding box information
            j["bbox"] = {
                {"x", bbox.x},
                {"y", bbox.y},
                {"width", bbox.width},
                {"height", bbox.height}
            };
            
            // Add timestamp
            auto now = std::chrono::system_clock::now();
            auto timestamp = std::chrono::system_clock::to_time_t(now);
            j["timestamp"] = std::ctime(&timestamp);
            
            // Write to file
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Unable to open file: " + filename);
            }
            
            file << std::setw(4) << j << std::endl;
            file.close();
        }
        catch (const std::exception& e) {
            std::cerr << "Error saving JSON: " << e.what() << std::endl;
            throw;
        }
    }
    
    static cv::Rect loadBoundingBox(const std::string& filename) {
        try {
            // Read JSON file
            std::ifstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Unable to open file: " + filename);
            }
            
            json j;
            file >> j;
            
            // Parse bounding box
            cv::Rect bbox;
            bbox.x = j["bbox"]["x"];
            bbox.y = j["bbox"]["y"];
            bbox.width = j["bbox"]["width"];
            bbox.height = j["bbox"]["height"];
            
            return bbox;
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading JSON: " << e.what() << std::endl;
            throw;
        }
    }
};