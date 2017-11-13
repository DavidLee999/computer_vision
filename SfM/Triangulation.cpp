//
//  Triangulation.cpp
//  sfm
//
//  Created by 李芃桦 on 2017/11/12.
//  Copyright © 2017年 李芃桦. All rights reserved.
//

#include "Triangulation.hpp"

#include <iostream>
using namespace std;

using namespace cv;

#undef __SFM__DEBUG__

Mat_<double> LinearLSTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1)
{
    Matx34d A (u.x * P(2, 0) - P(0, 0), u.x * P(2, 1) - P(0, 1), u.x * P(2, 2) - P(0, 2),
               u.y * P(2, 0) - P(1, 0), u.y * P(2, 1) - P(1, 1), u.y * P(2, 2) - P(1, 2),
               u1.x * P1(2, 0) - P1(0, 0), u1.x * P1(2, 1) - P1(0, 1), u1.x * P1(2, 2) - P1(0, 2),
               u1.y * P1(2, 0) - P1(1, 0), u1.y * P1(2, 1) - P1(1, 1), u1.y * P1(2, 2) - P1(1, 2)
               );
    
    Matx41d B (-(u.x * P(2, 3) - P(0, 3)),
               -(u.y * P(2, 3) - P(1, 3)),
               -(u1.x * P1(2, 3) - P1(0, 3)),
               -(u1.y * P1(2, 3) - P1(1, 3)));
    
    Mat_<double> X;
    cv::solve(A, B, X, DECOMP_SVD);
    
    return X;
}

Mat_<double> IterativeLinearLSTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1)
{
    double wi = 1, wi1 = 1;
    Mat_<double> X(4, 1);
    
    Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
    X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
    
    for (int i = 0; i < 10; ++i)
    {
        double p2x = Mat_<double>(Mat_<double>(P).row(2) * X)(0);
        double p2x1 = Mat_<double>(Mat_<double>(P1).row(2) * X)(0);
        
        if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON)
            break;
        
        wi = p2x;
        wi1 = p2x1;
        
        Matx34d A ((u.x * P(2, 0) - P(0, 0)) / wi, (u.x * P(2, 1) - P(0, 1)) / wi, (u.x * P(2, 2) - P(0, 2)) / wi,
                   (u.y * P(2, 0) - P(1, 0)) / wi, (u.y * P(2, 1) - P(1, 1)) / wi, (u.y * P(2, 2) - P(1, 2)) / wi,
                   (u1.x * P1(2, 0) - P1(0, 0)) / wi1, (u1.x * P1(2, 1) - P1(0, 1)) / wi1, (u1.x * P1(2, 2) - P1(0, 2)) / wi1,
                   (u1.y * P1(2, 0) - P1(1, 0)) / wi1, (u1.y * P1(2, 1) - P1(1, 1)) / wi1, (u1.y * P1(2, 2) - P1(1, 2)) / wi1
                   );
        
        Mat_<double> B = (Mat_<double>(4, 1) << -(u.x * P(2, 3) - P(0, 3)) / wi, - (u.y * P(2, 3) - P(1, 3)) / wi, -(u1.x * P1(2, 3) - P1(0, 3)) / wi1, -(u1.y * P1(2, 3) - P1(1, 3)) / wi1);
        
        cv::solve(A, B, X_, DECOMP_SVD);
        
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
    }
        
    return X;
}

double TriangulatePoints(const vector<KeyPoint>& pt_set1, const vector<KeyPoint>& pt_set2, const Mat& K, const Mat& Kinv, const Mat& distcoeff, const Matx34d& P, const Matx34d& P1, vector<CloudPoint>& pointcloud, vector<KeyPoint>& correspImg1Pt)
{
#ifdef __SFM__DEBUG__
    vector<double> depths;
#endif
    correspImg1Pt.clear();
    
    Matx44d P1_(P1(0, 0), P1(0, 1), P1(0, 2), P1(0, 3),
                P1(1, 0), P1(1, 1), P1(1, 2), P1(1, 3),
                P1(2, 0), P1(2, 1), P1(2, 2), P1(2, 3),
                0, 0, 0, 1);
    
    Matx44d P1inv = (P1_.inv());
    
    cout << "Triangulating...";
    double t = getTickCount();
    vector<double> reproj_error;
    unsigned int pts_size = pt_set1.size();
    
    Mat_<double> KP1 = K * Mat(P1);
#pragma omp parallel for num_threads(1)
    for (int i = 0; i < pts_size; ++i)
    {
        Point2f kp = pt_set1[i].pt;
        Point3d u (kp.x, kp.y, 1.0);
        Mat_<double> um = Kinv * Mat_<double>(u);
        u.x = um(0);
        u.y = um(1);
        u.z = um(2);
        
        Point2f kp1 = pt_set2[i].pt;
        Point3d u1 (kp1.x, kp1.y, 1.0);
        Mat_<double> um1 = Kinv * Mat_<double>(u1);
        u1.x = um1(0);
        u1.y = um1(1);
        u1.z = um1(2);
        
        Mat_<double> X = IterativeLinearLSTriangulation(u, P, u1, P1);
        
        Mat_<double> xPt_img = KP1 * X;
        
        Point2f xPt_img_ (xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));
#pragma omp critical
        {
            double reprj_err = norm(xPt_img_ - kp1);
            reproj_error.push_back(reprj_err);
            
            CloudPoint cp;
            cp.pt = Point3d(X(0), X(1), X(2));
            cp.reprojection_error = reprj_err;
            
            pointcloud.push_back(cp);
            correspImg1Pt.push_back(pt_set1[i]);
#ifdef __SFM_DEBUG__
            depths.push_back(X(2));
#endif
        }
    }
    
    Scalar mse = mean(reproj_error);
    t = ((double)getTickCount() - t) / getTickFrequency();
    
    cout << "Done. (" << pointcloud.size() << "points, " << t << "s, mean reproj_err = " << mse[0]<< ")" << endl;
    
#ifdef __SFM__DEBUG__
    {
        double minVal, maxVal;
        minMaxLoc(depths, &minVal, &maxVal);
        Mat tmp(240, 320, CV_8UC3, Scalar(0, 0, 0));
        
        for (unsigned int i = 0; i < pointcloud.size(); ++i)
        {
            double _d = MAX(MIN((pointcloud[i].z - minVal) / (maxVal - minVal), 1.0), 0.0);
            circle(tmp, correspImg1Pt[i].pt, 1, Scalar(255 * (1.0 - (_d)), 255, 255), CV_FILLED);
        }
        cvtColor(tmp, tmp, CV_HSV2BGR);
        imshow("Depth Map", tmp);
        waitKey(0);
        destroyWindow("Depth Map");
    }
#endif
    
    return mse[0];
}
