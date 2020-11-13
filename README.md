# Depth-aware Mixed Reality: Capture the AR-Flag

## Description

This repository contains the implementation for the project: Depth-aware Mixed Reality: Capture the AR-Flag. The project's goal is to augment reality with a virtual object which can be interacted with by multiple people. It is split into multiple parts which are listed and whose progress is displayed in [Progress](#Progress). This project was done in the context of the [lab course - perception and learning in robotics and augmented reality](http://campar.in.tum.de/Chair/TeachingSS20PLARR).

## Progress

- [x] Depth Estimation and Camera Pose
	- [x] Depth Map Filtering
	- [x] Kinect Fusion Integration
- [x] User Interaction and Scene Manipulation
	- [x] ChArUco Marker Tracking
	- [x] Human Pose Estimation
- [x] Realism and Embedding
	- [x] Augmented Rendering
	- [x] Image Harmonization
- [x] (Extension) Multiplayer Game: Capture the plane 

## Dependencies

The project depends on the following libraries:

- OpenCV with ArUco contrib module
- Caffe
- Pytorch
- Pyrender

## Project Demo
A video showcasing our implementation:

[![Demo of the implementation](https://img.youtube.com/vi/ajUJJ6zXu8w/0.jpg)](http://www.youtube.com/watch?v=ajUJJ6zXu8w "Demo of the implementation")

The intermediate slides presenting the project can be seen [here](docs/PLARR_5_mid_presentation.pdf).
For the final slides go [here](docs/PLARR_5_final_presentation.pdf).