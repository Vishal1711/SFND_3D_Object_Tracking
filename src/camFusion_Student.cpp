
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    //cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


void nextClosestLidarPoints(std::vector<LidarPoint> &lidarPoints, vector<double> &closestPoints, int n)
{
    for(int i = 0; i < n; ++i)
    {
        double minX = 1e9;
        for(auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it)
        {
            minX = ((minX > it->x) && (it->x > closestPoints[i])) ? it->x : minX;
        }
        closestPoints.push_back(minX);
    }
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // For estimating TTC, we need the closest reliable LiDAR point in both previous and current point clouds
    vector<double> closestPointsPrev, closestPointsCurr;  //for robust estimation to avoid considering outliers
    
    //2 Approches implemented - 1) considering 5 closest points. 2) considering all LiDAR points
    /* Approach 1 Begin - considering 5 closest points */ //Line 161-178
    // Finding closest distance in both previous and current lidar points
    double minXCurr = 1e9, minXPrev = 1e9;
    for(auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        minXPrev = minXPrev > it->x ? it->x : minXPrev;
    }

    for(auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        minXCurr = minXCurr > it->x ? it->x : minXCurr;
    }

    closestPointsPrev.push_back(minXPrev);
    closestPointsCurr.push_back(minXCurr);

    //minXPrev and minXCurr could be reliable closest LiDAR points or unwanted outliers
    //Finding next 4 closest points in both previous and current frames
    nextClosestLidarPoints(lidarPointsPrev, closestPointsPrev, 4); //by changing parameter 4 here, we can find as many closest points as required
    nextClosestLidarPoints(lidarPointsCurr, closestPointsCurr, 4);
    //Now we have 5 closest points in both previous and current frames
    /* Approach 1 end - considering 5 closest points */

    /* Approach 2 Begin - considering all LiDAR points */ //Line - 183-193
    /**for(auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        closestPointsPrev.push_back(it->x);
    }

    for(auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        closestPointsCurr.push_back(it->x);
    }
    sort(closestPointsPrev.begin(), closestPointsPrev.end());
    sort(closestPointsCurr.begin(), closestPointsCurr.end());**/
    //Now we have all lidar point distances sorted
    /* Approach 2 end - considering all LiDAR points */

    /* to check distance between closest LiDAR points to identify outliers based on distance thresold */
  	// distance thresold - if distance between 2 successive closest LiDAR points is greater than distance thresold then the closest point is an outlier
    /*cout << "closestPointsPrev ";
    for(int i = 0; i < 5 ; ++i){
        cout << "- " << i << ") " <<
        closestPointsPrev[i] << " ";
    }
    cout << endl;

    cout << "closestPointsCurr ";
    for(int i = 0; i < 5 ; ++i){
        cout << "- " << i << ") " <<
        closestPointsCurr[i] << " ";
    }
    cout << endl;

    cout << "Dist btwn Prev closest Points ";
    for(int i = 0; i < 4 ; ++i){
        cout << "- " << i << ") " <<
        closestPointsPrev[i+1] - closestPointsPrev[i] << " ";
    }
    cout << endl;

    cout << "Dist btwn Curr closest Points ";
    for(int i = 0; i < 4 ; ++i){
        cout << "- " << i << ") " <<
        closestPointsCurr[i+1] - closestPointsCurr[i] << " ";
    }
    cout << endl;*/
    /* trial */

    //checking which closest point to consider.
    //Threshold distance between 2 reliable closest points in a Point cloud is taken as 0.01m
    double closestXPrev, closestXCurr;
    
    /* Approach 1 begin */ //Inefficient approach - improved below
    /*if((closestPointsPrev[1] - closestPointsPrev[0]) < 0.03 && (closestPointsPrev[2] - closestPointsPrev[1])  < 0.03 &&
        (closestPointsPrev[3] - closestPointsPrev[2]) < 0.03)
    {
        closestXPrev = closestPointsPrev[0];
    } else if((closestPointsPrev[2] - closestPointsPrev[1]) < 0.03 && (closestPointsPrev[3] - closestPointsPrev[2])  < 0.03 &&
              (closestPointsPrev[4] - closestPointsPrev[3]) < 0.03)
    {
        closestXPrev = closestPointsPrev[1];
    }

    if((closestPointsCurr[1] - closestPointsCurr[0]) < 0.03 && (closestPointsCurr[2] - closestPointsCurr[1])  < 0.03 &&
        (closestPointsCurr[3] - closestPointsCurr[2]) < 0.03)
    {
        closestXCurr = closestPointsCurr[0];
    } else if((closestPointsCurr[2] - closestPointsCurr[1]) < 0.03 && (closestPointsCurr[3] - closestPointsCurr[2])  < 0.03 &&
              (closestPointsCurr[4] - closestPointsCurr[3]) < 0.03)
    {
        closestXCurr = closestPointsCurr[1];
    }*/
    /* Approach 1 end */

    /* Improvization 1 begin */
    /*for(size_t i = 0; i < closestPointsPrev.size() - 3; ++i)
    {
        if((closestPointsPrev[i+1] - closestPointsPrev[i]) < 0.03 && (closestPointsPrev[i+2] - closestPointsPrev[i+1])  < 0.03 &&
            (closestPointsPrev[i+3] - closestPointsPrev[i+2]) < 0.03)
        {
            closestXPrev = closestPointsPrev[i];
            break;
        } 
    }

    for(size_t i = 0; i < closestPointsCurr.size() - 3; ++i)
    {
        if((closestPointsCurr[i+1] - closestPointsCurr[i]) < 0.03 && (closestPointsCurr[i+2] - closestPointsCurr[i+1])  < 0.03 &&
            (closestPointsCurr[i+3] - closestPointsCurr[i+2]) < 0.03)
        {
            closestXCurr = closestPointsCurr[i];
            break;
        } 
    }*/
    /* Improvization 1 end */

    /* Improvization 2 begin */ //refactoring above improvisation by implementing bitwise operators
    for(size_t i = 0; i < closestPointsPrev.size() - 3; ++i)
    {
        bool check = true;
        for(int j = 0; j < 3; ++j)
        {
            check &= (closestPointsPrev[i+j+1] - closestPointsPrev[i+j]) < 0.01;
        }
        if(check)
        {
            closestXPrev = closestPointsPrev[i];
            break;
        } 
    }

    for(size_t i = 0; i < closestPointsCurr.size() - 3; ++i)
    {
        bool check = true;
        for(int j = 0; j < 3; ++j)
        {
            check &= (closestPointsCurr[i+j+1] - closestPointsCurr[i+j]) < 0.01;
        }
        if(check)
        {
            closestXCurr = closestPointsCurr[i];
            break;
        } 
    }
    /* Improvization 2 end */
    
    //TTC computation
    double dT = 1 / frameRate;
    //TTC = minXCurr * dT / (minXPrev - minXCurr);
    TTC = closestXCurr * dT / (closestXPrev - closestXCurr);

}


// associating bounding boxes between previous and current frame using keypoint matches
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    //multimap container for potential match candidates
    std::multimap<int, int> potentialBBMatches;

    //loop over keypoint matches
    for(std::vector<cv::DMatch>::iterator it1 = matches.begin(); it1 != matches.end(); ++it1)
    {
        // extracting matched keypoints from prev and curr frames
        cv::KeyPoint KptCurr = currFrame.keypoints.at(it1->trainIdx);
        cv::KeyPoint KptPrev = prevFrame.keypoints.at(it1->queryIdx);

        //checking which bounding boxes enclose these keypoints

        //to check if keypoint is enclosed in more than 1 bounding box
        vector<vector<BoundingBox>::iterator> KptCurrEnclosingBoxes, KptPrevEnclosingBoxes;

        //looping over Bounding boxes in current frame
        std::vector<BoundingBox> currFrameBBoxes = currFrame.boundingBoxes;
        
        //for(std::vector<BoundingBox>::iterator it2 = currFrameBBoxes.begin(); it2 != currFrameBBoxes.end(); ++it2)
        for(auto it2 = currFrameBBoxes.begin(); it2 != currFrameBBoxes.end(); ++it2)  //same as above line
        {
            if(it2->roi.contains(KptCurr.pt))
            {
                KptCurrEnclosingBoxes.push_back(it2);
            }
        }

        //looping over Bounding boxes in previous frame
        std::vector<BoundingBox> prevFrameBBoxes = prevFrame.boundingBoxes;
        for(auto it3 = prevFrameBBoxes.begin(); it3 != prevFrameBBoxes.end(); ++it3)
        {
            if(it3->roi.contains(KptPrev.pt))
            {
                KptPrevEnclosingBoxes.push_back(it3);
            }
        }

        //checking and ignoring if matched keypoint in either frame is enclosed by more than 1 bounding box or no bouding box
        if((KptCurrEnclosingBoxes.size() == 1) && (KptPrevEnclosingBoxes.size() == 1)) 
        {
            int prevBboxID = KptPrevEnclosingBoxes[0]->boxID;
            int currBboxID = KptCurrEnclosingBoxes[0]->boxID;
            
            //adding potential match candidates into a multimap container
            potentialBBMatches.insert(std::pair<int, int> (prevBboxID, currBboxID));
        }
    }

    ////checking for match candidates with highest number of occurences
    int prevBBoxesCount = prevFrame.boundingBoxes.size();

    //bounding box IDs in any DataFrame starts from 0
    for(size_t i = 0; i < prevBBoxesCount; ++i)
    {
        //equal_range() function returns a pair of iterators
        std::pair<multimap<int, int>::iterator, multimap<int, int>::iterator> it = potentialBBMatches.equal_range(i);
        vector<int> currBBIds; // Bounding boxes in current frame mapped to previous bounding box with boxID = i
        
        //using equal_range(i) to get potential matches of previous frame bouding box with ID = i. 
        //Bounding box IDs in any DataFrame starts from 0
        //if value 'i' is not present as key in multimap, then below for loop will not run, then currBBIds.size() = 0
        for(multimap<int, int>::iterator it1 = it.first; it1 != it.second; ++it1)
        {
            //collecting all mapped values to key 'i'
            currBBIds.push_back(it1->second);
        }

        if(currBBIds.size() > 0){ /// to avoid segmentation fault error for using 'currBBIds[0]' if currBBIds.size() = 0
            
            //finding value with highest number of occurences
            sort(currBBIds.begin(), currBBIds.end());
            int maxCount = 1, element = currBBIds[0], count = 1;
            for(int j = 1; j < currBBIds.size(); ++j)
            {
                if(currBBIds[j] == currBBIds[j-1]) {
                    count++;
                } else {
                    if(count > maxCount){
                        maxCount = count;
                        element = currBBIds[j-1];
                    }
                    count = 1;
                }
            }
            //if last element is the most frequent element
            if(count > maxCount){
                maxCount = count;
                element = currBBIds[currBBIds.size()-1];
            }

            // 'element' is the variable that holds highest occuring matching current frame bounding box ID
            //inserting best matches
            bbBestMatches.insert(pair<int, int>(i, element));
        }
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    vector<cv::DMatch> enclosedMatches;  ///Keypoint matches enclosed by region of interest of bounding box.

    for(auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        cv::KeyPoint kptCurr = kptsCurr[it->trainIdx];
        if(boundingBox.roi.contains(kptCurr.pt))
        {
            enclosedMatches.push_back(*it);
        }
    }

    //Problem is that there will be outliers among enclosedMatches
    //solution - computing mean of all euclidian distances between keypoint matches and then remove those that are too far away from the mean
    vector<double> distances;
    for(auto it = enclosedMatches.begin(); it != enclosedMatches.end(); ++it)
    {
        cv::KeyPoint kptCurr = kptsCurr[it->trainIdx];
        cv::KeyPoint kptPrev = kptsPrev[it->queryIdx];

        //euclidian distance between matched keypoints
        double eucDist = cv::norm(kptCurr.pt - kptPrev.pt);
        distances.push_back(eucDist);
    }

    //computing mean of all distances
    double meanDist = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();

    //cout << "------------------------------------------" << endl;
    //cout << "size of distances vector - " << distances.size() << endl;
    //cout << "mean of disatnces - " << meanDist << endl;
    //sort(distances.begin(), distances.end());

    /**for(size_t i = 0; i < distances.size(); ++i)
    {
        cout << i+1 << ") " << distances[i] << " - ";
    }
    cout << endl;**/

    //filtering collected keypoint matches based on euclidian distance between keypoint matches
    //filtering criterion is - euclidian distance between keypoint matches must fall between 1 and 3 times mean distance(3*meanDist). Decided by observing data.
    for(auto it = enclosedMatches.begin(); it != enclosedMatches.end(); ++it)
    {
        cv::KeyPoint kptCurr = kptsCurr[it->trainIdx];
        cv::KeyPoint kptPrev = kptsPrev[it->queryIdx];

        //euclidian distance between matched keypoints
        double eucDist = cv::norm(kptCurr.pt - kptPrev.pt);
        
        if(!((eucDist >= 1) && (eucDist < (3*meanDist))))
        {
            enclosedMatches.erase(it);
            it--;
        }
    }

    //adding filtered keypoint correspondences to the respective bounding box
    boundingBox.kptMatches = enclosedMatches;

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    //First sorting in  ascending order to finding the median
    sort(distRatios.begin(),distRatios.end());
    double medianDistRatio;

    //Finding median by considering both even and odd number of elements cases
    int numOfDistCalc = distRatios.size();
    if(numOfDistCalc%2 != 0){
       medianDistRatio = distRatios.at((numOfDistCalc-1)/2);
    } else {
        medianDistRatio = (distRatios.at((numOfDistCalc/2)-1) + distRatios.at(numOfDistCalc/2))/2;
    }

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);

}
