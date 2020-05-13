#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
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

#include "dataStructures.h"


void detectKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false,
                           bool verbose = false);

void detectKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false,
                              bool verbose = false);

void
detectKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, const std::string &detectorType,
                      bool bVis = false,
                      bool verbose = false);

void
describeKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors,
                  const std::string &descriptorType,
                  bool verbose);

void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef,
                      cv::Mat &descSource, cv::Mat &descRef, std::vector<cv::DMatch> &matches,
                      const std::string &descriptorType, const std::string &matcherType,
                      const std::string &selectorType,
                      bool verbose);

#endif /* matching2D_hpp */
