
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        //int normType = cv::NORM_HAMMING;
        int normType = (descriptorType.compare("DES_BINARY") == 0) ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching cross-check=" << crossCheck;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if(descSource.type() != CV_32F)
        {//OpenCV bug workaround: Convert binary descriptors into floating point decriptors due to bug in current version of OpenCV
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::FlannBasedMatcher::create();
        cout << "FLANN matching" << crossCheck;
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knnMatches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knnMatches, 2);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knnMatches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        float descDistRatio = 0.8;
        //Implementing Lowe's descriptor distance ratio test with threshold =0.8
        for(auto it = knnMatches.begin(); it != knnMatches.end(); ++it){
            //Now every 'it' is again a vector of cv::DMatches
            //Macthes are returned in distance increasing order. Hence first DMatch in 'it' is the best match
            if((*it)[0].distance < descDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout << "No of matches after Lowe's ratio test (KNN) is n=" << matches.size() << endl;
        cout << "# Keypoint macthes removed = " << knnMatches.size() - matches.size() << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    //vector<cv::KeyPoint> keypoints;  --> We already have keypoints as argument
    double maxOverlap = 0.0;
    for(size_t j = 0; j < dst_norm.rows; ++j)
    {
        for(size_t i = 0; i < dst_norm.cols; ++i){
            int response = (int)dst_norm.at<float>(j, i);
            if(response > minResponse){
                cv::KeyPoint newKeyPoint;

                //newKeyPoint.pt = cv::Point2f(j, i);
                //observe that above usage is wrong as x coordinate is along columns
                //and y coordinate is along rows. Hence use as below
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2*apertureSize;
                newKeyPoint.response = response;

                bool bOverlap = false;
                for(auto it = keypoints.begin(); it < keypoints.end(); ++it){
                    double overlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if(overlap > maxOverlap){
                        bOverlap = true;

                        if(response > (*it).response){
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }

                if(!bOverlap){
                    keypoints.push_back(newKeyPoint);
                }
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "HARRIS detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if(bVis){
        string windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName, 6);
        cv::Mat visImg = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImg);
        cv::waitKey(0);
    }
}

// Detect keypoints in image with various methods using OpenCV common interface wrapper
void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector;
    string windowName;

    if(detectorType.compare("FAST") == 0)
    {
        int threshold = 30;
        bool bNMS = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;

        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "FAST detector with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "FAST Detector Results";
    } 
    else if(detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "BRISK detector with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "BRISK Detector Results";
    } 
    else if(detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "ORB detector with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "ORB Detector Results";
    } 
    else if(detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "AKAZE detector with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "AKAZE detector Results";
    } 
    else if(detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "SIFT detector with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "SIFT detector Results";
    }

    if(bVis){
        cv::namedWindow(windowName, 6);
        cv::Mat visImg = img.clone();
        cv::drawKeypoints(img, keypoints, visImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImg);
        cv::waitKey(0);
    }
}
