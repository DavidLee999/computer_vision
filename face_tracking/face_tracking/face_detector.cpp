//
//  face_detector.cpp
//  face_tracking
//
//  Created by 李芃桦 on 2017/12/7.
//  Copyright © 2017年 李芃桦. All rights reserved.
//

#include "face_detector.hpp"
#include "ft.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

#define fl at<float>

vector<Point2f> face_detector::detect(const cv::Mat &im, const float scaleFactor, const int minNeighbours, const Size minSize)
{
    // convert image to grayscale
    Mat gray;
    if (im.channels() == 1)
        gray = im;
    else
        cvtColor(im, gray, CV_RGB2GRAY);
    
    // detect faces
    vector<Rect> faces;
    Mat eqIm;
    equalizeHist(gray, eqIm);
    detector.detectMultiScale(eqIm, faces, scaleFactor, minNeighbours, 0 | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, minSize);
    
    if (faces.size() < 1)
        return vector<Point2f>();
    
    // predict face placement
    Rect R = faces[0];
    Vec3f scale = detector_offset * R.width;
    int n = reference.rows / 2;
    vector<Point2f> p(n);
    
    for (int i = 0; i < n; ++i)
    {
        p[i].x = scale[2] * reference.fl(2 * i) + R.x + 0.5 * R.width + scale[0];
        p[i].y = scale[2] * reference.fl(2 * i + 1) + R.y + 0.5 + R.height + scale[1];
    }
    
    return p;
}

void face_detector::train(ft_data &data, const string fname, const cv::Mat &ref, const bool mirror, const bool visi, const float frac, const float scaleFactor, const int minNeighbours, const Size minSize)
{
    detector.load(fname.c_str());
    detector_fname = fname;
    reference = ref.clone();
    vector<float> xoffset(0), yoffset(0), zoffset(0);
    
    for (int i = 0; i < data.n_images(); ++i)
    {
        Mat im = data.get_image(i, 0);
        if (im.empty())
            continue;
        
        vector<Point2f> p = data.get_points(i, false);
        int n = p.size();
        
        Mat pt = Mat(p).reshape(1, 2 * n);
        
        vector<Rect> faces;
        Mat eqIm;
        equalizeHist(im, eqIm);
        
        detector.detectMultiScale(eqIm, faces, scaleFactor, minNeighbours, 0 | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, minSize);
        
        t;
    }
}
