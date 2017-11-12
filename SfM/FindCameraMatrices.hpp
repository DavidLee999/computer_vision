//
//  FindCameraMatrices.hpp
//  sfm
//
//  Created by 李芃桦 on 2017/11/12.
//  Copyright © 2017年 李芃桦. All rights reserved.
//

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Common.hpp"

bool CheckCoherentRotation(cv::Mat_<double>& R);

bool TestTriangulation(const std::vector<CloudPoint>& pcloud, const cv::Matx34d& P, std::vector<uchar>& status);

cv::Mat GetFundamentalMat(const std::vector<cv::KeyPoint>& imgpts1, const std::vector<cv::KeyPoint>& imgpts2, std::vector<cv::KeyPoint>& imgpts1_good, std::vector<cv::KeyPoint>& imgpts2_good, std::vector<cv::DMatch>& matches
#ifdef __SFM__DEBUG__
                          ,const cv::Mat& = cv::Mat(), const cv::Mat& = cv::Mat()
#endif
);

bool FindCameraMatrices(const cv::Mat& K, const cv::Mat& Kinv, const cv::Mat& distcoeff, const std::vector<cv::KeyPoint>& imgpts1, const std::vector<cv::KeyPoint>& imgpts2, std::vector<cv::KeyPoint>& imgpts1_good, std::vector<cv::KeyPoint>& imgpts2_good, cv::Matx34d& P, cv::Matx34d& P1, std::vector<cv::DMatch>& matches, std::vector<CloudPoint>& outCloud
#ifdef __SFM__DEBUG__
                        ,const cv::Mat& = cv::Mat(), const cv::Mat& = cv::Mat()
#endif
);
