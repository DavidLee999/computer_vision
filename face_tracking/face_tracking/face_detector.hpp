#ifndef face_detector_hpp
#define face_detector_hpp

#include "ft_data.cpp"

#include <opencv2/objdetect/objdetect.hpp>

class face_detector { // face detector for initializetion
public:
    string detector_fname; // file containing cascade classifier
    Vec3f detector_offset; // offset from center of detection
    Mat reference; // reference shape
    CascadeClassifier detector; // face detector
    
    vector<Point2f> detect(const Mat& im, // image containing face
                           const float scaleFactor = 1.1, // scale increment
                           const int minNeighbours = 2, // minmum neighbourhood size
                           const Size minSize = Size(30, 30)); // minimum detection window size
    
    void train(ft_data& data, // training data
               const string fname, // cascade detector
               const Mat& ref, // reference shape
               const bool mirror = false, // mirror data?
               const bool visi = false, // visualise training?
               const float frac = 0.8, // fraction of points in detected rect
               const float scaleFactor = 1.1, // scale increment
               const int minNeighbours = 2, // minimum neighbourhood size
               const Size minSize = Size(30, 30)); // minimum detection window size
    
    void write(FileStorage& fs) const;
    
    void read (const FileNode& node);
    
protected:
    bool enough_bounded_points(const Mat& pts, // points to evaluate
                               const Rect R, // bounding rectangle
                               const float frac); // fraction of points bounded
    
    Point2f center_of_mass(const Mat& pts); // [x1;x2;...;xn;yn]
    
    float calc_scale(const Mat& pts); // scaling from @reference to @pts
};

#endif /* face_detector_hpp */
