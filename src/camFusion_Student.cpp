
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"


namespace {

    template<typename type>
    int ifloor(type value) {
        return static_cast<int>(std::floor(value));
    }

    using box_id_type = decltype(BoundingBox::boxID);

    std::vector<box_id_type>
    findBoxIdsContainingKeypoint(const std::vector<BoundingBox> &boxes, const cv::KeyPoint &keyPoint) {
        const auto numBoxes = boxes.size();
        const auto prevPoint = pointFromKeypoint(keyPoint);

        std::vector<box_id_type> boxIds;
        for (auto b = 0; b < numBoxes; ++b) {
            const auto &box = boxes[b];
            if (!box.roi.contains(prevPoint)) continue;
            boxIds.push_back(box.boxID);
        }

        return boxIds;
    }

    LidarPoint getMedianPointByXCoordinate(std::vector<LidarPoint> &points) {
        std::sort(points.begin(), points.end(),
                  [](const LidarPoint &pt1, const LidarPoint &pt2) {
                      return pt1.x < pt2.x;
                  });

        const auto medianIndex = points.size() / 2;

        const auto oddNumberOfPoints = (points.size() & 1U) == 1;
        assert((points.size() % 2 != 0) == oddNumberOfPoints);

        if (oddNumberOfPoints) {
            return points[medianIndex];
        }

        const auto leftMedian = points[medianIndex - 1];
        const auto rightMedian = points[medianIndex];

        const auto x = (leftMedian.x + rightMedian.x) * 0.5;
        const auto y = (leftMedian.y + rightMedian.y) * 0.5;
        const auto z = (leftMedian.z + rightMedian.z) * 0.5;
        const auto r = (leftMedian.r + rightMedian.r) * 0.5;
        return {x, y, z, r};
    }

    double containedPointMeanDistance(const BoundingBox &boundingBox, const std::vector<cv::DMatch> &matches,
                                      const std::vector<cv::KeyPoint> &kptsPrev,
                                      const std::vector<cv::KeyPoint> &kptsCurr) {
        auto ssd = 0.0;
        auto count = 0.0;
        for (const auto &match : matches) {
            const auto currPt = kptsCurr[match.trainIdx];
            const auto prevPt = kptsPrev[match.queryIdx];
            if (!boundingBox.roi.contains(currPt.pt)) continue;

            ssd += cv::norm(currPt.pt - prevPt.pt);
            count += 1.0;
        }

        return ssd / count;
    }

    // https://stackoverflow.com/a/253874/195651
    bool essentiallyEqual(double a, double b, double epsilon) {
        return std::abs(a - b) <= ((std::abs(a) > std::abs(b) ? std::abs(b) : std::abs(a)) * epsilon);
    }

}


cv::Point pointFromKeypoint(const cv::KeyPoint &keyPoint) {
    return {ifloor(keyPoint.pt.x), ifloor(keyPoint.pt.y)};
}


// Create groups of LiDAR points whose projection into the camera falls into the same bounding box
void
clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor,
                    cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT) {
    // loop over all LiDAR points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (const auto &lidarPoint : lidarPoints) {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = lidarPoint.x;
        X.at<double>(1, 0) = lidarPoint.y;
        X.at<double>(2, 0) = lidarPoint.z;
        X.at<double>(3, 0) = 1;

        // project LiDAR point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        std::vector<std::vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current LiDAR point
        for (auto it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check whether point is within current bounding box
            if (smallerBox.contains(pt)) {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check whether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1) {
            // add LiDAR point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(lidarPoint);
        }

    } // eof loop over all LiDAR points
}


void
show3DObjects(const std::string &tag, std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize,
              bool bWait) {
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (const auto &box : boundingBoxes) {
        // create randomized color for current 3D object
        cv::RNG rng(box.boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot LiDAR points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (const auto &lidarPoint : box.lidarPoints) {
            // world coordinates
            float xw = lidarPoint.x; // world position in m with x facing forward from sensor
            float yw = lidarPoint.y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", box.boxID, (int) box.lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i) {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    std::string windowName = tag + " 3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_GUI_NORMAL);
    cv::imshow(windowName, topviewImg);

    if (bWait) {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches) {

    const double averageDistance = containedPointMeanDistance(boundingBox, kptMatches, kptsPrev, kptsCurr);
    const double tolerance = averageDistance * 1.25;

    for (const auto &match : kptMatches) {
        const auto currPt = kptsCurr[match.trainIdx];
        const auto prevPt = kptsPrev[match.queryIdx];
        if (!boundingBox.roi.contains(currPt.pt)) continue;

        const auto d = cv::norm(currPt.pt - prevPt.pt);
        if (d >= tolerance) continue;

        boundingBox.keypoints.push_back(currPt);
        boundingBox.kptMatches.push_back(match);
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg) {

    // If there are no keypoints, there are no ratios, and thus there is no velocity to be calculated.
    // Likewise, we need at least two correspondences to determine a relative distance change.
    if (kptMatches.size() < 2) {
        TTC = NAN;
        return;
    }

    // Some constants used for checking in order to prevent
    // division of too small numbers by too big numbers, or dividing
    // by zero altogether.
    const auto minDistPx = 100.0; // minimum required distance in pixels
    const auto epsilon = 1e-4;

    // Assuming observed distances in images are independent of direction,
    // we can determine the change of keypoint distance between frames
    // to determine a change in scale and by that a change in physical distance.
    std::vector<double> distanceRatios;

    // For each keypoint pair
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) {
        const auto currKpToken = kptsCurr.at(it1->trainIdx);
        const auto prevKpToken = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) {
            const auto currKp = kptsCurr.at(it2->trainIdx);
            const auto prevKp = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            const auto distCurr = cv::norm(currKpToken.pt - currKp.pt);
            const auto distPrev = cv::norm(prevKpToken.pt - prevKp.pt);

            // avoid division by zero
            if (distPrev <= epsilon || distCurr < minDistPx) continue;

            const auto distRatio = distCurr / distPrev;
            distanceRatios.push_back(distRatio);
        }
    }

    assert(!distanceRatios.empty());

    // Use the median distance ratio to obtain a stable value.
    const auto medianIndex = distanceRatios.size() / 2;
    std::sort(distanceRatios.begin(), distanceRatios.end());
    const auto distanceRatio = distanceRatios.size() % 2 == 0
                               ? (distanceRatios[medianIndex - 1] + distanceRatios[medianIndex]) / 2.0
                               : distanceRatios[medianIndex];

    // Note that if the median distance ratio is 1 (e.g. because no keypoints change),
    // then no velocity can be observed and the function below would result in a division by zero.
    // In this case, it takes infinity seconds to collide - so we make it explicit here.
    if (essentiallyEqual(distanceRatio, 1.0, 1e-6)) {
        TTC = std::numeric_limits<double>::infinity();
        return;
    }

    const auto dT = 1 / frameRate;
    TTC = -dT / (1 - distanceRatio);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {

    // Our main assumption is that we can find a stable estimate of the distance
    // to our car by obtaining the median of all points along the X axis.
    // Note that in our setup, the X direction is "forward" and thus
    // extends into the depth of the image.

    const auto medianPre = getMedianPointByXCoordinate(lidarPointsPrev);
    const auto prevX = medianPre.x;

    const auto medianCurr = getMedianPointByXCoordinate(lidarPointsCurr);
    const auto currX = medianCurr.x;

    // We can now use a constant velocity model to calculate an
    // estimate for TTC. Note that in this consideration,
    // d(t + ΔT) < d(t) if the car in front of ego is braking.
    // Another relevant assumption here is that we keep a constant ΔT
    // between frames.

    // Given that
    //      TTC = d1 / v_0
    // and
    //      d(t + ΔT) = d(t) - v0 * ΔT
    //      v0 = ( d(t) - d(t + ΔT) ) / ΔT
    //      v0 = (  d0  -     d1    ) / ΔT
    // we find by substitution that
    //      TTC = ΔT * d1 / ( d0 - d1)

    const auto dT = 1 / frameRate;
    TTC = dT * currX / (prevX - currX);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                        DataFrame &currFrame) {

    // For each match, determine all box correspondences between two frames.
    using prev_box_id_type = box_id_type;
    using curr_box_id_type = box_id_type;
    using keypoint_count_type = std::size_t;
    std::map<prev_box_id_type, std::map<curr_box_id_type, keypoint_count_type>> correspondences;

    // Fore each correspondence, register which box combination has the highest number of matched keypoints.
    // This will provide a stable match even if the same keypoint is contained in multiple boxes
    // in either the previous OR the current frame.
    for (const auto &match : matches) {
        // Sneaky one: Note the order of query and train indexes.
        const auto &prevKpt = prevFrame.keypoints[match.queryIdx];
        const auto &currKpt = currFrame.keypoints[match.trainIdx];

        const auto prevBoxIds = findBoxIdsContainingKeypoint(prevFrame.boundingBoxes, prevKpt);
        if (prevBoxIds.empty()) continue;

        const auto currBoxIds = findBoxIdsContainingKeypoint(currFrame.boundingBoxes, currKpt);
        if (currBoxIds.empty()) continue;

        // For each box pair, register the fact that the current keypoint was matched in both.
        // Ideally, we have found exactly one pair, corresponding to the same box across both frames.
        for (const auto prevBoxId : prevBoxIds) {
            for (const auto currBoxId : currBoxIds) {
                correspondences[prevBoxId][currBoxId] += 1;
            }
        }
    }

    // Link up the ID of the box in the previous frame
    // with the ID of the box with the highest number of matching keypoints in this frame.
    for (const auto &pair : correspondences) {
        const auto &prevBoxId = pair.first;
        assert(!pair.second.empty());

        using pair_type = decltype(pair.second)::value_type;
        const auto &currBoxMaxMatches = std::max_element(
                pair.second.begin(), pair.second.end(),
                [](const pair_type &entry1, const pair_type &entry2) {
                    return entry1.second < entry2.second;
                });

        bbBestMatches[prevBoxId] = currBoxMaxMatches->first;
    }
}
