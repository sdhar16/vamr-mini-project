# vamr-mini-project
This repository implements a simple, monocular, visual odometry (VO) pipeline with the most essential features: initialization of 3D landmarks, keypoint tracking between two frames, pose estimation using established 2D ↔ 3D correspondences, and triangulation of new landmarks.

You can run the repo for three datasets available in a folder named data.

If you are linux or macOs user, you can directly run the pipeline for three data using bash files. Just do the following:

To run on malaga dataset, do the following:
```
chmod +x malaga.sh
./malaga.sh
```
To run on parking dataset, do the following:
```
chmod +x parking.sh
./parking.sh
```
To run on KITTI dataset, do the following:
```
chmod +x kitti.sh
./kitti.sh
```
In case you wanna run for a dataset multiple times, you do not need to run chmod command multiple times. 

If you are windows user, you can still run the pipeline for all three dataset. But in order to do so, you need to open src/main/main.py file and uncomment the config of desired dataset.

To run on parking, just uncomment the parking config path.
```
CONFIG_PATH = "src/main/configs/Parking.yaml"
#CONFIG_PATH = "src/main/configs/malaga.yaml"
#CONFIG_PATH = "src/main/configs/KITTI.yaml" 
```
To run on malaga, just uncomment the malaga config path.
```
#CONFIG_PATH = "src/main/configs/Parking.yaml"
CONFIG_PATH = "src/main/configs/malaga.yaml"
#CONFIG_PATH = "src/main/configs/KITTI.yaml" 
```
To run on kitti, just uncomment the kitti config path.
```
#CONFIG_PATH = "src/main/configs/Parking.yaml"
#CONFIG_PATH = "src/main/configs/malaga.yaml"
CONFIG_PATH = "src/main/configs/KITTI.yaml" 
```
Once you uncomment the config path of desired dataset to run. Save the main file and simply run the main file as follows.

```
python3 src/main/main.py
```
Please note that we used following versions while developing:
1. conda 4.10.3
2. Python 3.8.8

In addition, we provided output of conda env export in a txt file conda_env.txt.