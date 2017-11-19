//
//  shape_model.hpp
//  face_tracking
//
//  Created by 李芃桦 on 2017/11/18.
//  Copyright © 2017年 李芃桦. All rights reserved.
//

#ifndef shape_model_hpp
#define shape_model_hpp

#include <opencv2/core/core.hpp>
using namespace cv;

#include <vector>
using namespace std;

class shape_model{
public:
    Mat p; // parameter vector (k x 1)
    Mat V; // shape basis (2n x k)
    Mat e; // parameter variance (k x 1)
    Mat C; // connectivity (c x 2)
    
    int npts()
    { return V.rows / 2; }
    
    void calc_params(const vector<Point2f>& pts, const Mat weight = Mat(), const float c_factor = 3.0);
    
    vector<Point2f> calc_shape(); // shape described by p
    
    void set_identity_params();
    
    Mat rot_scale_align(const Mat& src, const Mat& dst); // scaled rotation mat
    
    Mat center_shape(const Mat& pts); // centered shape
    
    void train(const vector<vector<Point2f> >& p, const vector<Vec2i>& con = vector<Vec2i>(), const float frac = 0.95, const int kmax = 10);
    
    void write(FileStorage& fs) const;
    
    void read(const FileNode& node);
    
protected:
    void clamp(const float c = 3.0);
    
    Mat pts2mat(const vector<vector<Point2f> >& p);
    
    Mat procrustes(const Mat& X, const int itol = 100, const float ftol = 1e-6);
    
    Mat calc_rigid_basis(const Mat& X);
};

#endif /* shape_model_hpp */
