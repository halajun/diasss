# A Fully-automatic Side-scan Sonar SLAM Framework

This repository presents a feature-based SLAM framework using side-scan sonar, which is able to automatically detect and robustly match keypoints between paired side-scan images and use them as constraints to optimize the AUV pose trajectory.

# 1. License

This code is released under a [GPLv3 license](https://github.com/halajun/diasss/blob/main/LICENSE).

If you use this code in an academic work, please cite:

    @article{zhang2023rsn,
    title={A fully-automatic side-scan sonar simultaneous localization and mapping framework},
    author={Zhang, Jun and Xie, Yiping and Ling, Li and Folkesson, John},
    journal={IET Radar, Sonar \& Navigation},
    year={2023},
    publisher={Wiley Online Library}
    }

# 2. Prerequisites
We have tested the library in **Ubuntu 20.04.5**, but it should be easy to compile in other platforms. 

## c++11, gcc 
We use some functionalities of c++11, and the tested gcc version is 9.4.0.

## Boost
Download and install instructions can be found at: https://www.boost.org/. **Required at least 1.65.0**.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Download and install instructions can be found at: http://opencv.org. **Tested with OpenCV 4.6**.

## Eigen3
Required by GTSAM (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 2.91.0**.

## GTSAM
We use [gtsam](https://gtsam.org/) library to perform non-linear optimizations. **Tested with GTSAM 4.2**

# 3. Building Library

Clone the repository:
```
git clone https://github.com/halajun/diasss.git diasss
```

We provide a `CMakeLists.txt` to build the libraries. 
Please make sure you have installed all required dependencies (see section 2). 

Then Execute:
```
cd diasss
mkdir build
cd build
cmake ..
make
```

This will create 

1. **libdiasss.so** at *build* folder,

2. and the executable **test_demo** in *bin* folder of the source path.

# 4. Running Pre-processed Data

The command to run the executable file is:

```
../bin/./test_demo --image ../test_data/img-xml/ --pose ../test_data/pose-xml/ --altitude ../test_data/altitude/ --groundrange ../test_data/groundrange/ --annotation ../test_data/annos-xml/
```
Here:
1. --image: the canonical transformed (or normal, should also work) side-scan image is saved in xml file of OpenCV (using the function cv::FileStorage()), in the size of NxM;
2. --pose: the initial pose (dead-reckoning) is also saved in xml file, in size of Nx6 (roll,pitch,yaw,x,y,z).
3. --altitude： the altitude of AUV is saved in txt， in the size of Nx1；
4. --groundrange: the ground range of side-scan image is also saved in txt, in the size of (M/2)x1;
5. --annotation: the ground truth annotation of keypoints (used for evaluation) is saved as xml file, in the size of Kx7 (source_img_id, target_img_id, u_source,v_source, u_target, v_target, prior_depth), and K is the number of keypoint correspondences. One could modify the code to ignored the annotation input if there are no annotations available.

# 5. Canonical Representation for Side-scan Image

The details of cannonical transformation for side-scan image can be found in [SSS Canonical Representation](https://github.com/luxiya01/SSS-Canonical-Representation). If you use the code of canonical transformation in an academic work, please cite:

    @INPROCEEDINGS{xu2023oceans,
    author={Xu, Weiqi and Ling, Li and Xie, Yiping and Zhang, Jun and Folkesson, John},
    booktitle={OCEANS 2023 - Limerick}, 
    title={Evaluation of a Canonical Image Representation for Sidescan Sonar}, 
    year={2023},
    pages={1-7},
    doi={10.1109/OCEANSLimerick52467.2023.10244293}}


