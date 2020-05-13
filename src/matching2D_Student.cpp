#include "matching2D.hpp"


// Descriptor extractors (Task MP.2)
namespace {

    cv::Ptr<cv::FeatureDetector> createFastDetector() {
        const auto threshold = 30;
        const auto nonmaxSuppression = true;
        const auto type = cv::FastFeatureDetector::TYPE_9_16;
        return cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type);
    }

    cv::Ptr<cv::FeatureDetector> createBriskDetector() {
        const auto threshold = 30;
        const auto octaves = 3;
        const auto patternScale = 1.0F;
        return cv::BRISK::create(threshold, octaves, patternScale);
    }

    cv::Ptr<cv::FeatureDetector> createOrbDetector() {
        // https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
        const auto numFeatures = 500;
        const float scaleFactor = 1.2F;
        const auto numLevels = 8;
        const auto edgeThreshold = 31;
        const auto firstLevel = 0;
        const auto WTA_K = 2;
        const auto scoreType = cv::ORB::HARRIS_SCORE;
        const auto patchSize = 31;
        const auto fastThreshold = 20;
        return cv::ORB::create(numFeatures, scaleFactor, numLevels, edgeThreshold, firstLevel, WTA_K, scoreType,
                               patchSize, fastThreshold);
    }

    cv::Ptr<cv::FeatureDetector> createAkazeDetector() {
        const auto descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
        const auto descriptorSize = 0;
        const auto descriptorChannels = 3;
        const auto threshold = 0.001f;
        const auto numOctaves = 4;
        const auto numOctaveLayers = 4;
        const auto diffusivity = cv::KAZE::DIFF_PM_G2;
        return cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels,
                                 threshold, numOctaves, numOctaveLayers, diffusivity);
    }

    cv::Ptr<cv::FeatureDetector> createSiftDetector() {
        const auto numFeatures = 0;
        const auto numOctaveLayers = 3;
        const auto contrastThreshold = 0.04;
        const auto edgeThreshold = 10;
        const auto sigma = 1.6;
        return cv::xfeatures2d::SIFT::create(numFeatures, numOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }

    cv::Ptr<cv::FeatureDetector> createDetector(const std::string &detectorType) {
        if (detectorType == "FAST") {
            return createFastDetector();
        } else if (detectorType == "BRISK") {
            return createBriskDetector();
        } else if (detectorType == "ORB") {
            return createOrbDetector();
        } else if (detectorType == "AKAZE") {
            return createAkazeDetector();
        } else if (detectorType == "SIFT") {
            return createSiftDetector();
        }

        return nullptr;
    }

}

// Descriptor extractors (Task MP.4)
namespace {

    cv::Ptr<cv::DescriptorExtractor> createBriskDescriptorExtractor() {
        const auto threshold = 30;        // FAST/AGAST detection threshold score.
        const auto octaves = 3;           // detection octaves (use 0 to do single scale)
        const auto patternScale = 1.0F;   // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        return cv::BRISK::create(threshold, octaves, patternScale);
    }

    cv::Ptr<cv::DescriptorExtractor> createBriefDescriptorExtractor() {
        const auto bytes = 32;
        const auto use_orientation = false;
        return cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    }

    cv::Ptr<cv::DescriptorExtractor> createOrbDescriptorExtractor() {
        // The ORB detector also is an extractor.
        return createOrbDetector();
    }

    cv::Ptr<cv::DescriptorExtractor> createFreakDescriptorExtractor() {
        const auto orientationNormalized = true;
        const auto scaleNormalized = true;
        const auto patternScale = 22.0F;
        const auto nOctaves = 4;
        return cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
    }

    cv::Ptr<cv::DescriptorExtractor> createAkazeDescriptorExtractor() {
        // The AKAZE detector also is an extractor.
        return createAkazeDetector();
    }

    cv::Ptr<cv::DescriptorExtractor> createSiftDescriptorExtractor() {
        // The SIFT detector also is an extractor.
        return createSiftDetector();
    }

    cv::Ptr<cv::DescriptorExtractor> createDescriptorExtractor(const std::string &detectorType) {
        if (detectorType == "BRISK") {
            return createBriskDescriptorExtractor();
        } else if (detectorType == "BRIEF") {
            return createBriefDescriptorExtractor();
        } else if (detectorType == "ORB") {
            return createOrbDescriptorExtractor();
        } else if (detectorType == "FREAK") {
            return createFreakDescriptorExtractor();
        } else if (detectorType == "AKAZE") {
            return createAkazeDescriptorExtractor();
        } else if (detectorType == "SIFT") {
            return createSiftDescriptorExtractor();
        }

        return nullptr;
    }

}

namespace {

    cv::Ptr<cv::DescriptorMatcher> createBruteForceMatcher(bool crossCheck, bool isBinaryDescriptor) {
        const auto normType = isBinaryDescriptor ? cv::NORM_HAMMING : cv::NORM_L2SQR;
        return cv::BFMatcher::create(normType, crossCheck);
    }

    cv::Ptr<cv::DescriptorMatcher> createFlannBasedMatcher(bool isBinaryDescriptor) {
        if (isBinaryDescriptor) {
            return new cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        } else {
            return cv::FlannBasedMatcher::create();
        }
    }

    cv::Ptr<cv::DescriptorMatcher>
    createDescriptorMatcher(const std::string &matcherType, bool crossCheck, bool isBinaryDescriptor) {
        if (matcherType == "MAT_BF") {
            return createBruteForceMatcher(crossCheck, isBinaryDescriptor);
        } else if (matcherType == "MAT_FLANN") {
            return createFlannBasedMatcher(isBinaryDescriptor);
        }

        return nullptr;
    }

}

namespace {

    void
    matchKeypointsNearestNeighbor(const cv::Ptr<cv::DescriptorMatcher> &matcher, cv::Mat &descSource, cv::Mat &descRef,
                                  std::vector<cv::DMatch> &matches) {
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }

    void
    matchKeypointsKNearestNeighbor(const cv::Ptr<cv::DescriptorMatcher> &matcher, cv::Mat &descSource, cv::Mat &descRef,
                                   std::vector<cv::DMatch> &matches, bool verbose) {
        // Specifically, k = 2.
        std::vector<std::vector<cv::DMatch>> nearestNeighbors;
        matcher->knnMatch(descSource, descRef, nearestNeighbors, 2);

        // Apply distance ratio test.
        const auto minDescDistRatio = 0.8; // Lowe's magic number
        for (auto &pair : nearestNeighbors) {
            if (pair.size() < 2) {
                continue;
            } else if (pair[0].distance < (minDescDistRatio * pair[1].distance)) {
                matches.push_back(pair[0]);
            }
        }

        if (verbose) {
            std::cout << "Kept " << matches.size() << " / " << nearestNeighbors.size() << " keypoints" << std::endl;
        }
    }

    void
    matchKeypoints(const std::string &selectorType, const cv::Ptr<cv::DescriptorMatcher> &matcher, cv::Mat &descSource,
                   cv::Mat &descRef, std::vector<cv::DMatch> &matches, bool verbose) {
        if (selectorType == "SEL_NN") { // nearest neighbor (best match)
            matchKeypointsNearestNeighbor(matcher, descSource, descRef, matches);
            return;
        } else if (selectorType == "SEL_KNN") { // k nearest neighbors (k=2)
            matchKeypointsKNearestNeighbor(matcher, descSource, descRef, matches, verbose);
            return;
        }

        assert(false);
    }

}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef,
                      cv::Mat &descSource, cv::Mat &descRef, std::vector<cv::DMatch> &matches,
                      const std::string &descriptorType, const std::string &matcherType,
                      const std::string &selectorType, bool verbose) {
    // configure matcher
    const auto crossCheck = false;
    const auto isBinaryDescriptor = descriptorType == "DES_BINARY";
    const auto matcher = createDescriptorMatcher(matcherType, crossCheck, isBinaryDescriptor);

    // perform matching task
    matchKeypoints(selectorType, matcher, descSource, descRef, matches, verbose);
}


// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void describeKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors,
                       const std::string &descriptorType, bool verbose) {
    const auto extractor = createDescriptorExtractor(descriptorType);
    assert(extractor != nullptr);

    extractor->compute(img, keypoints, descriptors);
}


// Detect keypoints in image using the traditional Shi-Thomasi detector
void detectKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, bool verbose) {
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / std::max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (const auto &corner : corners) {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(corner.x, corner.y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }

    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detectKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, bool verbose) {
    const auto blockSize = 4;
    const auto apertureSize = 3;
    const auto k = 0.04;
    const auto minResponse = 0.3F;  // minimum response required to keep a point
    const auto maxOverlap = 0.F;    // maximum overlap to "neighboring" keypoints; [0 .. 1]

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k);

    cv::Mat norm, normScaled;
    cv::normalize(dst, norm, 0, 1.0F, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // Adjust for visualization.
    convertScaleAbs(norm, normScaled);

    // To apply non-maxima suppression, we march over every pixel and determine the
    // detector response at each location. For each nontrivial response we first
    // generate a candidate keypoint and then determine whether we should keep or discard it.
    for (auto y = 0; y < norm.rows; ++y) {
        const auto *row = norm.ptr<float>(y);
        for (auto x = 0; x < norm.cols; ++x) {
            const auto response = row[x];
            if (response <= minResponse) {
                continue;
            }

            cv::KeyPoint kp{cv::Point2f(x, y), 2 * apertureSize};
            kp.response = response;

            // We now compare the existing set of keypoints for overlaps with the candidate's
            // region in order to keep only the point with the highest response.
            auto registerKeypoint = true;

            // Note that this is a super naive implementation, but it get's the point across.
            // An indexing structure such as a Kd- or Quadtree could be used to speed up the process a bit.
            // We could also
            for (auto i = 0; i < keypoints.size(); ++i) {
                const auto &keypoint = keypoints[i];
                const auto overlapAmount = cv::KeyPoint::overlap(kp, keypoint);
                if (overlapAmount <= maxOverlap) {
                    continue;
                }

                // We have an overlap - either of both keypoints will be dropped,
                // so no new point is added.
                registerKeypoint = false;

                if (kp.response > keypoint.response) {
                    keypoints[i] = kp;
                    break;
                }
            }

            if (registerKeypoint) {
                keypoints.push_back(kp);
            }
        }
    }

    if (bVis) {
        std::string windowName = "Harris detector results";
        cv::namedWindow(windowName);
        cv::imshow(windowName, normScaled);
        cv::waitKey(0);
    }
}

void
detectKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, const std::string &detectorType, bool bVis, bool verbose) {
    const auto detector = createDetector(detectorType);
    assert(detector != nullptr);

    detector->detect(img, keypoints);

    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        const auto windowName = detectorType + " detector results";

        cv::namedWindow(windowName);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
