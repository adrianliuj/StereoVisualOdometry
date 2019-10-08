# StereoVisualOdometry
This library is a feature-based visual odometry that is designed to run on KITTI dataset. It uses ORB to extract 
key points and generate descriptors. It then triangulates matching points to get depth and support 3d-3d (ICP) 
and 3d-2d (PNP) pose estimation. It also uses RANSAC to remove outliers.

## 1. Dependencies
This library is tested on KITTI dataset on ubuntu 16.04.
### Pangolin
Pangolin is used for visualize vehicle trajectory (maybe for world points later)
### OpenCV
Tested on OpenCV 3.4.3. Other versions should be fine.
### Eigen3
Used for linear calculation, rotation and translation represetation, etc.

## 2. Build Instruction
In linux environment, pop-up the terminal and enter following commands:
```
git clone https://github.com/adrianliuj/StereoVisualOdometry.git
cd StereoVisualOdometry
mkdir build
cd build
cmake ..
make
```
## 3. Run on KITTI dataset
Execute by entering following command:
```
./run_vo PATH_TO_KITTI_SEQUENCE
```
for example, on my machine, I run
```
./run_vo /media/ubuntu/drive1/dataset/sequences/02
```
