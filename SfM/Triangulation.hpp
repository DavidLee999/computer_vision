//
//  Triangulation.hpp
//  sfm
//
//  Created by 李芃桦 on 2017/11/12.
//  Copyright © 2017年 李芃桦. All rights reserved.
//

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#ifdef __SFM__DEBUG__
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include <vector>

#include "Common.hpp"

cv::Mat_<double> LinearLSTriangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1);

#define EPSILON 0.0001

cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1);

double TriangulatePoints(const std::vector<cv::KeyPoint>& pt_set1, const std::vector<cv::KeyPoint>& pt_set2, const cv::Mat& K, const cv::Mat& diffcoeff, const cv::Matx34d& P, const cv::Matx34d& P1, std::vector<CloudPoint>& pointcloud, std::vector<cv::KeyPoint>& cprrespImg1Pt);
