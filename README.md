# 🚀 Hybrid Object Tracking with YOLOv11, CSRT, and Distractor-Aware Memory (DAM)

This project combines deep learning–based object detection (YOLOv11 via NCNN) with real-time CSRT tracking and a distractor-aware memory (DAM) system to achieve robust object tracking under challenging conditions such as occlusion, clutter, and camera motion. The pipeline runs CSRT for fast local tracking and periodically uses YOLOv11 to correct drift and update the tracker.

To handle occlusions, the system maintains two memory buffers: Recent Appearance Memory (RAM) stores recent confirmed detections, while Distractor Resolving Memory (DRM) tracks visually stable distractors. If the object is lost, DAM intelligently reinitializes tracking using these memories in a prioritized order, ensuring robust recovery.
