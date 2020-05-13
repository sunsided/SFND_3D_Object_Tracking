# SFND 3D Object Tracking

Welcome to the final project of the camera course.
By completing all the lessons, you now have a solid understanding of
keypoint detectors, descriptors, and methods to match them between
successive images. Also, you know how to detect objects in an image
using the YOLO deep-learning framework. And finally, you know how to
associate regions in a camera image with LiDAR points in 3D space.
Let's take a look at our program schematic to see what we already have
accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic.
To do this, you will complete four major tasks: 

1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on LiDAR measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or LiDAR sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

### Dependencies for Running Locally

* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Matching 3D Objects

First, regions of interests are obtained in `detectObjects()`
using a [YOLO](https://pjreddie.com/darknet/yolo/) v3
detector that was trained on the [COCO](http://cocodataset.org/) ("_Common Objects in Context_")
data set. The COCO dataset contains a couple of classes relevant to street scene
understanding, such as

- person, 
- bicycle, 
- car, motorbike, bus, truck,
- traffic light, stop sign

and more. Since a pre-trained network was used, detections were trimmed
after the fact to only provide `car` class instances, and all predictions
with a confidence lower than 0.2 were discarded. This yielded the following
result:

![](.readme/yolo-cars.png)

Next, the LiDAR a point cloud obtained in `loadLidarFromFile()`
from a [Velodyne HDL-64E](http://velodynelidar.com/lidar/hdlproducts/hdl64e.aspx) 
sensor (data was given as part of the [KITTI](http://www.cvlibs.net/datasets/kitti/) dataset)
was cropped in `cropLidarPoints()` to contain only points that lie

- within the ego lane,
- about 2 m behind up to 20 m ahead of the car, and
- roughly above road surface.

A planar road without lateral curvature was assumed for simplicity.
After this, all LiDAR not lying within one of the previously detected
rectangular regions of interest (ROI) were discarded in `clusterLidarWithROI()`.
To counter rough ROI shapes and oversized boxes (and artifact of the neural
network's output), each detected rectangle was reduced by approximately 20%
along width and height. The bounding boxes were then shrinked to exactly contain
all LiDAR points. Due to the previous cropping, only one box is obtained as a result:

![](.readme/lidar-boxed.png)

During this process, each LiDAR point is also associated with the bounding box it is confined in.

In order to perform image feature detection, a mask was obtained in `createKeypointMask()`
from the ROIs to focus only on relevant areas of the image (the cars, as far as this
project is concerned):

![](.readme/keypoint-mask.png)

Note that due to the limitation on the `car` class, the truck on the
right side of the image is effectively invisible after this procedure.
This is acceptable in the setup of this project.

A [FAST](https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test)
feature detector using BRIEF keypoints was selected (note that this
combination almost resembles [ORB](https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF),
which was however discarded as part of the conclusions of the [2D Feature Tracking](https://github.com/sunsided/SFND_2D_Feature_Tracking)
project) and used to obtain keypoints in `detectKeypointsModern()`
and provide descriptors in `describeKeypoints()`.

Keypoints were then matched with the previous frame in `matchDescriptors()` ...

![](.readme/keypoints-fast-brief.png)

... in order to determine ROI / bounding box correspondences across frames in `matchBoundingBoxes()`.
By this, tracking of a 2D object through time is implemented.

Since LiDAR points were already associated with bounding boxes as
part of the initial steps, correspondence between 3D measurements from
the LiDAR point cloud and linked 2D information obtained from the YOLO
detector can be obtained, which effectively allows tracking 3D objects
(the LiDAR points) over time:

![](.readme/lidar-projected-boxed.png)

Note that at this point of the project, a Time-to-Collision estimate
was not yet generated, resulting in a reported `0.000000 s` value in the
referenced picture.

## Computing LiDAR-based Time-to-Collision

Due to the measurements of physical distance of the car in front
of the ego car we can now utilize changes in distance to approximate
velocity and, by extension, time-to-collision.  

Under the assumption of a constant velocity model as an approximation 
of the velocity during small time steps ΔT, and by making use of
the clustered LiDAR points of the car in front of the ego car,
we can utilize the change of the median X ("front") distances between
frames to provide a TTC estimate in `computeTTCLidar()`:

```cpp
const auto dT = 1 / frameRate;
const auto TTC = dT * currX / (prevX - currX);
```
