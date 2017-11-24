//
//  ft_data.hpp
//  face_tracking
//
//  Created by 李芃桦 on 2017/11/24.
//  Copyright © 2017年 李芃桦. All rights reserved.
//

#ifndef ft_data_hpp
#define ft_data_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#include <vector>
using namespace std;

class ft_data{
public:
    vector<int> symmetry; // indices of symmetric points
    vector<Vec2i> connections; // indices of connected points
    vector<string> imnames; // images
    vector<vector<Point2f> > points;
    
    inline int n_images()
    { return imnames.size(); }
    
    Mat get_image(const int idx, const int flag = 2); // get idx's image. gray = 0, grap + flip = 1, rgb = 2, rgb + flip = 3
    
    vector<Point2f> get_points(const int idx, const bool filpped = false); // points for idx's image
    
    void rm_incomplete_samples(); // remove samples having missing point annotations
    
    void rm_sample(const int idx);
    
    void draw_points(Mat& im, // image to draw on
                     const int idx, // index of shape
                     const bool flipped = false, // flip points?
                     const Scalar color = CV_RGB(255, 0, 0), // color of points
                     const vector<int>& pts = vector<int>()); // indces of points to draw
    
    void draw_sym(Mat& im, // image to draw on
                  const int idx, // index of shape
                  const bool flipped = false, // flip points?
                  const vector<int>& pts = vector<int>()); // indices of points to draw
    
    void draw_connect(Mat& im, // image to draw on
                      const int idx, // index of shape
                      const bool flipped = false, // flip points?
                      const Scalar color = CV_RGB(0, 0, 255), // color
                      const vector<int>& con = vector<int>()); // indices of connections
    
    void write(FileStorage& fs) const;
    
    void read(const FileNode& node);
};

#endif /* ft_data_hpp */
