//
//  face_tracker.hpp
//  face_tracking
//
//  Created by 李芃桦 on 2017/12/10.
//  Copyright © 2017年 李芃桦. All rights reserved.
//

#ifndef face_tracker_hpp
#define face_tracker_hpp

#include "patch_model.hpp"
#include "shape_model.hpp"
#include "face_detector.hpp"
#include "fps_timer.h"

class face_tracker_params { // face tracking parameters
public:
    vector<Size> ssize; // search region size/level
    bool robust; // use robust fitting?
    int itol; // maximum number of iterations to try
    float ftol; // vonvergence tolerance
    float scaleFactor; // OpenCV cascade detector parameters
    int minNeighbours;
    Size minSize;
    
    face_tracker_params();
    
    void write(FileStorage& fs) const;
    
    void read(const FileNode& node);
};

class face_tracker { // face tracking class
public:
    bool tracking; // are we in tracking mode
    fps_timer timer; // frames/second timer
    vector<Point2f> points; // current tracked points
    face_detector detector; // detector for initialization
    shape_model smodel; // shape model
    patch_models pmodel; // feature detectors
    
    face_tracker()
    { tracking = false; }
    
    int track(const Mat& im, // 0: failure. image containing face
              const face_tracker_params& p = face_tracker_params()); // fitting parameters
    
    void reset() // reset tracker
    {
        tracking = false;
        timer.reset();
    }
    
    void draw(Mat& im,
              const Scalar pts_color = CV_RGB(255, 0, 0),
              const Scalar con_color = CV_RGB(0, 255, 0));
    
    void write(FileStorage& fs) const;
    
    void read(const FileNode& node);
    
protected:
    vector<Point2f> // points for fitted face in image
    fit(const Mat& image, // image containing face
        const vector<Point2f>& init, // initial point estimates
        const Size ssize = Size(21, 21), // search region size
        const bool robust = false, // use robust fitting
        const int itol = 10, // maximum number of iterations to try
        const float ftol = 1e-3); // convergence tolerance
};

#endif /* face_tracker_hpp */
