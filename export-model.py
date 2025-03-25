from ultralytics import YOLO

yolo11_small = "/Users/3i-a1-2022-062/workspace/yolo11_track/best.pt"
model = YOLO(yolo11_small)


# model.export(
#     format='coreml', 
#     nms=True,  # Non-Maximum Suppression
#     imgsz=640,  # Image size
# )

# Export the model to NCNN format
model.export(format="ncnn")  # creates '/yolo11n_ncnn_model'
