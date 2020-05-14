
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <deque>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

// #define VERBOSE


void visualizeMatches(const std::_Deque_iterator<DataFrame, DataFrame &, DataFrame *>::_Self &currentDataBuffer,
                      const std::_Deque_iterator<DataFrame, DataFrame &, DataFrame *>::_Self &previousDataBuffer,
                      const std::vector<cv::DMatch> &matches);

void dropKeypointsNotWithinAnyBoundingBox(
        const std::_Deque_iterator<DataFrame, DataFrame &, DataFrame *>::_Self &currentDataBuffer,
        std::vector<cv::KeyPoint> &keypoints);

cv::Mat
createKeypointMask(const DataFrame &currentDataBuffer);

BoundingBox *getBoxById(std::vector<BoundingBox> &boxes, const int &boxId);

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    std::string dataPath = "../";

    // camera
    std::string imgBasePath = dataPath + "images/";
    std::string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    std::string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes LiDAR and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1;
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    std::string yoloBasePath = dataPath + "dat/yolo/";
    std::string yoloClassesFile = yoloBasePath + "coco.names";
    std::string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    std::string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // LiDAR
    std::string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    std::string lidarFileType = ".bin";

    // calibration data for camera and LiDAR
    cv::Mat P_rect_00(3, 4, cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4, 4, cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4, 4, cv::DataType<double>::type); // rotation matrix and translation vector

    RT.at<double>(0, 0) = 7.533745e-03;
    RT.at<double>(0, 1) = -9.999714e-01;
    RT.at<double>(0, 2) = -6.166020e-04;
    RT.at<double>(0, 3) = -4.069766e-03;
    RT.at<double>(1, 0) = 1.480249e-02;
    RT.at<double>(1, 1) = 7.280733e-04;
    RT.at<double>(1, 2) = -9.998902e-01;
    RT.at<double>(1, 3) = -7.631618e-02;
    RT.at<double>(2, 0) = 9.998621e-01;
    RT.at<double>(2, 1) = 7.523790e-03;
    RT.at<double>(2, 2) = 1.480755e-02;
    RT.at<double>(2, 3) = -2.717806e-01;
    RT.at<double>(3, 0) = 0.0;
    RT.at<double>(3, 1) = 0.0;
    RT.at<double>(3, 2) = 0.0;
    RT.at<double>(3, 3) = 1.0;

    R_rect_00.at<double>(0, 0) = 9.999239e-01;
    R_rect_00.at<double>(0, 1) = 9.837760e-03;
    R_rect_00.at<double>(0, 2) = -7.445048e-03;
    R_rect_00.at<double>(0, 3) = 0.0;
    R_rect_00.at<double>(1, 0) = -9.869795e-03;
    R_rect_00.at<double>(1, 1) = 9.999421e-01;
    R_rect_00.at<double>(1, 2) = -4.278459e-03;
    R_rect_00.at<double>(1, 3) = 0.0;
    R_rect_00.at<double>(2, 0) = 7.402527e-03;
    R_rect_00.at<double>(2, 1) = 4.351614e-03;
    R_rect_00.at<double>(2, 2) = 9.999631e-01;
    R_rect_00.at<double>(2, 3) = 0.0;
    R_rect_00.at<double>(3, 0) = 0;
    R_rect_00.at<double>(3, 1) = 0;
    R_rect_00.at<double>(3, 2) = 0;
    R_rect_00.at<double>(3, 3) = 1;

    P_rect_00.at<double>(0, 0) = 7.215377e+02;
    P_rect_00.at<double>(0, 1) = 0.000000e+00;
    P_rect_00.at<double>(0, 2) = 6.095593e+02;
    P_rect_00.at<double>(0, 3) = 0.000000e+00;
    P_rect_00.at<double>(1, 0) = 0.000000e+00;
    P_rect_00.at<double>(1, 1) = 7.215377e+02;
    P_rect_00.at<double>(1, 2) = 1.728540e+02;
    P_rect_00.at<double>(1, 3) = 0.000000e+00;
    P_rect_00.at<double>(2, 0) = 0.000000e+00;
    P_rect_00.at<double>(2, 1) = 0.000000e+00;
    P_rect_00.at<double>(2, 2) = 1.000000e+00;
    P_rect_00.at<double>(2, 3) = 0.000000e+00;

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for LiDAR and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    std::deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += imgStepWidth) {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
        std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);
        if (dataBuffer.size() > dataBufferSize) {
            dataBuffer.pop_front();
        }

        // Helpers gonna help.
        auto currentDataBuffer = (dataBuffer.end() - 1);
        auto previousDataBuffer = (dataBuffer.end() - 2);
        const auto hasPreviousDataBuffer = dataBuffer.size() > 1;

#ifdef VERBOSE
        std::cout << "#1 : LOAD IMAGE INTO BUFFER done" << std::endl;
#endif


        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;
        detectObjects(currentDataBuffer->cameraImg, currentDataBuffer->boundingBoxes, confThreshold,
                      nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

#ifdef VERBOSE
        std::cout << "#2 : DETECT & CLASSIFY OBJECTS done" << std::endl;
#endif

        /* CROP LiDAR POINTS */

        // load 3D LiDAR points from file
        std::string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove LiDAR points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
        currentDataBuffer->lidarPoints = lidarPoints;

#ifdef VERBOSE
        std::cout << "#3 : CROP LiDAR POINTS done" << std::endl;
#endif

        /* CLUSTER LiDAR POINT CLOUD */

        // associate LiDAR points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI(currentDataBuffer->boundingBoxes, currentDataBuffer->lidarPoints, shrinkFactor,
                            P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        bVis = false;
        if (true) {
            if (hasPreviousDataBuffer) {
                show3DObjects("Previous", previousDataBuffer->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000),
                              false);
            }
            show3DObjects("Current", currentDataBuffer->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), false);
        }
        bVis = false;

#ifdef VERBOSE
        std::cout << "#4 : CLUSTER LiDAR POINT CLOUD done" << std::endl;
#endif

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor(currentDataBuffer->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // We will only allow keypoints to be found within the bounding boxes.
        const auto keypointMask = createKeypointMask(*currentDataBuffer);

#if 0
        cv::namedWindow("Keypoint Mask");
        cv::imshow("Keypoint Mask", keypointMask);
        cv::waitKey(0);
#endif

        // extract 2D keypoints from current image
        std::vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        std::string detectorType = "AKAZE";

        if (detectorType == "SHITOMASI") {
            detectKeypointsShiTomasi(keypoints, imgGray, keypointMask, false, false);
        } else if (detectorType == "HARRIS") {
            detectKeypointsHarris(keypoints, imgGray, keypointMask, bVis, false);
        } else {
            detectKeypointsModern(keypoints, imgGray, keypointMask, detectorType, bVis, false);
        }

#if 0
        // discard all keypoints that are not within any bounding box
        dropKeypointsNotWithinAnyBoundingBox(currentDataBuffer, keypoints);
#endif

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts) {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") ==
                0) { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            std::cout << " NOTE: Keypoints have been limited!" << std::endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        currentDataBuffer->keypoints = keypoints;

#ifdef VERBOSE
        std::cout << "#5 : DETECT KEYPOINTS done" << std::endl;
#endif

        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        std::string descriptorType = "AKAZE"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        describeKeypoints(currentDataBuffer->keypoints, currentDataBuffer->cameraImg, descriptors,
                          descriptorType, false);

        // push descriptors for current frame to end of data buffer
        currentDataBuffer->descriptors = descriptors;

#ifdef VERBOSE
        std::cout << "#6 : EXTRACT DESCRIPTORS done" << std::endl;
#endif

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            std::vector<cv::DMatch> matches;
            std::string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
            std::string descriptorMetaType = "DES_BINARY"; // DES_BINARY, DES_HOG
            std::string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            matchDescriptors(previousDataBuffer->keypoints, currentDataBuffer->keypoints,
                             previousDataBuffer->descriptors, currentDataBuffer->descriptors,
                             matches, descriptorMetaType, matcherType, selectorType, false);

            // store matches in current data frame
            currentDataBuffer->kptMatches = matches;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis && hasPreviousDataBuffer) {
                visualizeMatches(currentDataBuffer, previousDataBuffer, matches);

            }
            bVis = false;

#ifdef VERBOSE
            std::cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << std::endl;
#endif

            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)

            // associate bounding boxes between current and previous frame using keypoint matches
            std::map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *previousDataBuffer, *currentDataBuffer);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            currentDataBuffer->bbMatches = bbBestMatches;

#ifdef VERBOSE
            std::cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << std::endl;
#endif

            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (auto it1 = currentDataBuffer->bbMatches.begin();
                 it1 != currentDataBuffer->bbMatches.end(); ++it1) {

                // find bounding boxes associates with current match
                auto *currBB = getBoxById(currentDataBuffer->boundingBoxes, it1->second);
                auto *prevBB = getBoxById(previousDataBuffer->boundingBoxes, it1->first);

#if 0
                std::cout << "Matched box #" << prevBB->boxID
                          << " (" << prevBB->lidarPoints.size() << " LiDAR points)"
                          << " with #"
                          << currBB->boxID
                          << " (" << currBB->lidarPoints.size() << " LiDAR points)"
                          << std::endl;
#endif

                // compute TTC for current match
                if (!currBB->lidarPoints.empty() &&
                    !prevBB->lidarPoints.empty()) // only compute TTC if we have LiDAR points
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on LiDAR data (implement -> computeTTCLidar)
                    double ttcLidar;
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;
                    clusterKptMatchesWithROI(*currBB, previousDataBuffer->keypoints,
                                             currentDataBuffer->keypoints, currentDataBuffer->kptMatches);
                    computeTTCCamera(previousDataBuffer->keypoints, currentDataBuffer->keypoints,
                                     currBB->kptMatches, sensorFrameRate, ttcCamera);
                    //// EOF STUDENT ASSIGNMENT

                    std::cout << "#" << imgIndex << " "
                              << detectorType << " + " << descriptorType
                              << ": TTC LiDAR=" << ttcLidar
                              << ", TTC camera=" << ttcCamera
                              << std::endl;

                    bVis = false;
                    if (bVis) {
                        cv::Mat visImg = currentDataBuffer->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y),
                                      cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height),
                                      cv::Scalar(0, 255, 0), 2);

                        char str[200];
                        sprintf(str, "TTC LiDAR : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));

                        std::string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        std::cout << "Press key to continue to next frame" << std::endl;
                        cv::waitKey(0);
                    }
                    bVis = false;

                } // eof TTC computation
            } // eof loop over all BB matches

        }

    } // eof loop over all images

    return 0;
}

BoundingBox* getBoxById(std::vector<BoundingBox> &boxes, const int &boxId) {
    BoundingBox *currBB;
    for (auto &box : boxes) {
        if (boxId == box.boxID) { // check whether current match partner corresponds to this BB
            currBB = &box;
            break;
        }
    }
    return currBB;
}

cv::Mat
createKeypointMask(const DataFrame &currentDataBuffer) {
    const auto &srcImg = currentDataBuffer.cameraImg;
    cv::Mat keypointMask(srcImg.size[0], srcImg.size[1], CV_8UC1);
    keypointMask.setTo(0);
    for (const auto &box : currentDataBuffer.boundingBoxes) {
        cv::rectangle(keypointMask, box.roi, cv::Scalar(255, 255, 255), -1);
    }
    return keypointMask;
}

void dropKeypointsNotWithinAnyBoundingBox(
        const std::_Deque_iterator<DataFrame, DataFrame &, DataFrame *>::_Self &currentDataBuffer,
        std::vector<cv::KeyPoint> &keypoints) {
    keypoints.erase(
            std::remove_if(keypoints.begin(), keypoints.end(),
                           [&currentDataBuffer](const cv::KeyPoint &kpt) {
                               const auto &boxes = currentDataBuffer->boundingBoxes;
                               const auto point = pointFromKeypoint(kpt);
                               for (const auto &box : boxes) {
                                   // Don't delete if the point is within some box.
                                   if (box.roi.contains(point)) {
                                       return false;
                                   }
                               }
                               return true;
                           }),
            keypoints.end());
}

void visualizeMatches(const std::_Deque_iterator<DataFrame, DataFrame &, DataFrame *>::_Self &currentDataBuffer,
                      const std::_Deque_iterator<DataFrame, DataFrame &, DataFrame *>::_Self &previousDataBuffer,
                      const std::vector<cv::DMatch> &matches) {
    const auto drawFlags =
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS |
            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS;

    cv::Mat cloneImg = (currentDataBuffer->cameraImg).clone();
    cv::Mat matchImg = (currentDataBuffer->cameraImg).clone();
    cv::cvtColor(cloneImg, matchImg, cv::COLOR_BGR2GRAY);

    cv::drawMatches(previousDataBuffer->cameraImg, previousDataBuffer->keypoints,
                    currentDataBuffer->cameraImg, currentDataBuffer->keypoints,
                    matches, matchImg,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), drawFlags);

    std::string windowName = "Matching keypoints between two camera images";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, matchImg);
}

