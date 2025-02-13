
#include <iostream>
#include "detector_yolo_inference.hpp"
#include "utils.h"
#include "bounding_box_saver.h"

std::vector<std::string> modelDirectories = {
    "/models/pretrain/yolo11s_ncnn_model",
    "/models/pretrain/yolo11l_ncnn_model",
    "/models/custom/horse_only_yolo11s_2080ti_1_ver13_20250106_ncnn_model"
};

std::vector<std::string> modelFileName = {
    "/model.ncnn",
    "/model.ncnn",
    "/horse-only-yolo11"
};

int main(int argc, char *argv[]) {
    if (argc < 2) {
        // ./objdet <input> <param> <bin> 
        std::cerr << "Usage: " << argv[0] << "<proj_dir> <image_path> <output_path> <param> <bin>" << std::endl;
        return EXIT_FAILURE;
    }
    std::size_t modelID = 0;
    std::string projDir = argv[1];
    std::string modelDir = projDir + modelDirectories[modelID];
    std::string paramPath = modelDir + modelFileName[modelID] + ".param";
    std::string binPath = modelDir + modelFileName[modelID] + ".bin";


    std::string image_file = argv[2];
    std::string outputImgPath = argv[3];
    
    if (argc == 6) {
        paramPath = std::string(argv[4]);
        binPath = std::string(argv[5]);
    } else if (argc == 5) {
        std::cerr << "Please input both proto(param) and model(bin) file paths." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "<Model Path> " << std::endl;
    std::cout << "\tparamPath: " << binPath << std::endl;
    std::cout << "\tbinPath: " << binPath << std::endl;

    ncnn::Net yolov11;
    yolov11.load_param(paramPath.c_str());
    yolov11.load_model(binPath.c_str());

    // save
    std::string outputDir = filePath::getParentDirectory(outputImgPath);
    std::string outputJsonPath = outputDir + "/" + filePath::getStem(outputImgPath) + ".json";
    std::cout << "<Output Path> " << std::endl;
    std::cout << "\tOut Directory: " << outputDir << std::endl;
    std::cout << "\tOut Img Path : " << outputImgPath << std::endl;
    std::cout << "\tOut Json Path: " << outputJsonPath << std::endl;

    int numberOfClasses = 80;
    std::unordered_set<int> targetLabels = {0};
    DetectorClassInfo classInfo = {numberOfClasses, targetLabels};

    cv::Mat image = cv::imread(image_file);
    cv::Rect boxRoI(0,0,0,0);
    
    // inferenceOD->detect(image, boxRoI);
    std::vector<BoxInfo> detections;
    yolo::detect(classInfo, yolov11, image, detections);

    if (detections.size() > 0) {
        BBox detBox = detections[0].getBox();
        boxRoI = cv::Rect(detBox.x, detBox.y, detBox.w, detBox.h);
   
    	// save
    	BoundingBoxSaver::saveBoundingBox(outputJsonPath, boxRoI, image_file);
    	cv::Mat croppedImage = image(boxRoI);  // Extract the region of interest
    
    	std::cout << ">> BoxROI : " << boxRoI.x << ", " << boxRoI.y << ", " << boxRoI.width << ", " << boxRoI.height << std::endl;
    	std::cout << outputJsonPath << std::endl;
    	std::cout << outputImgPath << std::endl;
    	cv::imwrite(outputImgPath, croppedImage); // Save the cropped image
    }
    else {
	    std::cout << "No Target Detected!" << std::endl;
    }
    return 0;
}
