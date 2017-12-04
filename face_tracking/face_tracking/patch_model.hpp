//
//  patch_model.hpp
//  face_tracking
//
//  Created by 李芃桦 on 2017/12/1.
//  Copyright © 2017年 李芃桦. All rights reserved.
//

#ifndef patch_model_hpp
#define patch_model_hpp

#include "ft_data.hpp"

#include <opencv2/core/core.hpp>
using namespace cv;

#include <vector>
using namespace std;

class patch_model { // correlation-based patch expert
public:
    Mat P; // normalised patch
    
    inline Size patch_size() { return P.size(); }
    
    Mat calc_response(const Mat& im, const bool sum2one = false); // calc response maps
    
    void train(const vector<Mat>& images, // feature centered training images
               const Size psize, // desired path size
               const float var = 1.0, // variance of annotation error
               const float lambda = 1e-6, // regularization weight
               const float mu_init = 1e-3, // initial stoch-grad step size
               const int nsamples = 1000, // number of stoch-grad samples,
               const bool visi = false); // visualize intermediate results?
    
    void write(FileStorage& fs) const;
    
    void read(const FileNode& node);
    
protected:
    Mat convert_image(const Mat& im); // convert unsigned char image to single channel log-scale img.
    
};

class patch_models {
public:
    Mat reference; // reference shape
    vector<patch_model> patches; // patch model
    
    inline int n_patches() { return patches.size(); }
    
    void train(ft_data& data, // training data
               const vector<Point2f>& ref, // reference shape
               const Size psize, // desired patch size
               const Size ssize, // search window size
               const bool mirror = true, // use mirrored images?
               const float var = 1.0, // variance of annotation error
               const float lambda = 1e-6, // regularization weight
               const float mu_init = 1e-3, // initial soch-grad step size
               const int nsamples = 1000, // number of stoch-grad samples
               const bool visi = true); // visualize intermediate results?
    
    vector<Point2f> calc_peaks(const Mat& im, // image to detect features in
                               const vector<Point2f>& points, // initial estimate of shape
                               const Size ssize = Size(21, 21)); // search window size
    
    void write(FileStorage& fs) const;
    
    void read(const FileNode& node);
    
protected:
    Mat inv_simil(const Mat& S); // inverted similarity transform
    
    Mat calc_simil(const Mat& pts);  // similarity transform reference->pts
    
    vector<Point2f> // similarity trainsfromed shape
    apply_simil(const Mat& S, // similarity transform
                                const vector<Point2f>& points); // shape to transform
};
#endif /* patch_model_hpp */
