//
//  Common.cpp
//  sfm
//
//  Created by 李芃桦 on 2017/11/12.
//  Copyright © 2017年 李芃桦. All rights reserved.
//

#include "Common.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#include <iostream>
using namespace std;

#ifndef WIN32
#include <dirent.h>
#endif

std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches)
{
    std::vector<cv::DMatch> flip;
    for (int i = 0; i < matches.size(); ++i)
    {
        flip.push_back(matches[i]);
        swap(flip.back().queryIdx, flip.back().trainIdx);
    }
    
    return flip;
}

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts)
{
    std::vector<cv::Point3d> out;
    for (unsigned int i = 0; i < cpts.size(); ++i)
        out.push_back(cpts[i].pt);
    
    return out;
}

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1, const std::vector<cv::KeyPoint>& imgpts2, const std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& pt_set1, std::vector<cv::KeyPoint>& pt_set2)
{
    for (unsigned int i = 0; i < matches.size(); ++i)
    {
        assert(matches[i].queryIdx < imgpts1.size());
        pt_set1.push_back(imgpts1[matches[i].queryIdx]);
        
        assert(matches[i].trainIdx < imgpts2.size());
        pt_set2.push_back(imgpts2[matches[i].trainIdx]);
    }
}

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps)
{
    ps.clear();
    for (unsigned int i = 0; i < kps.size(); ++i)
        ps.push_back(kps[i].pt);
}

void PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps)
{
    kps.clear();
    for (unsigned int i = 0; i < ps.size(); ++i)
        kps.push_back(KeyPoint(ps[i], 1.0f));
}

#define intrpmnmx(val, min, max) (max == min ? 0.0 : ((val) - min) / (max - min))

void drawArrows(Mat& frame, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status, const vector<float>& verror, const Scalar& _line_color)
{
    double minVal, maxVal;
    minMaxIdx(verror, &minVal, &maxVal, 0, 0, status);
    int line_thickness = 1;
    
    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (status[i])
        {
            double alpha = intrpmnmx(verror[i], minVal, maxVal);
            alpha = 1 - alpha;
            Scalar line_color(alpha * _line_color[0], alpha * _line_color[1], alpha * _line_color[2]);
            
            Point p = prevPts[i];
            Point q = nextPts[i];
            
            double angle = atan2((double)p.y - q.y, (double) p.x - q.x);
            
            double hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + double(p.x - q.x) * (p.x - q.x));
            
            if (hypotenuse < 1.0)
                continue;
            
            q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
            q.y = (int)(p.y - 3 * hypotenuse * sin(angle));
            line(frame, p, q, line_color, line_thickness);
            
            p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
            
            p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length())
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    else
        return false;
}

bool hasEndingLower(string const &fullString_, string const &_ending)
{
    string fullString = fullString_, ending = _ending;
    transform(fullString_.begin(), fullString_.end(), fullString.begin(), ::tolower);
    
    return hasEnding(fullString, ending);
}

void imshow_250x250(const string& name_, const Mat& patch)
{
    Mat bigPatch;
    cv::resize(patch, bigPatch, Size (250, 250));
    imshow(name_, bigPatch);
}
