//
//  fps_timer.h
//  face_tracking
//
//  Created by 李芃桦 on 2017/12/10.
//  Copyright © 2017年 李芃桦. All rights reserved.
//

#pragma once

#include <iostream>

#include <cstdio>

class fps_timer { // frames/second timer for tracking
public:
    int64 t_start; // start time
    int64 t_end; // end time
    float fps; // current frames/sce
    int fnum; // number of frames since @t_start
    
    fps_timer()
    { this->reset(); }
    
    void increment() // increment timer index
    {
        if (fnum >= 29)
        {
            t_end = cv::getTickCount();
            fps = 30.0 / (float(t_end - t_start) / getTickFrequency());
            t_start = t_end;
            fnum = 0;
        }
        else
            fnum += 1;
    }
    
    void reset() // reset timer
    {
        t_start = cv::getTickCount();
        fps = 0;
        fnum = 0;
    }
    
    void display_fps(Mat& im, Point p = Point(-1, -1)) // image to display FPS on
    {
        char str[256];
        Point pt;
        if (p.y < 0)
            pt = Point(10, im.rows - 20);
        else
            pt = p;
        
        sprintf(str, "%d frames/sec", (int)cvRound(fps));
        string text = str;
        putText(im, test, pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255));
    }
    
};
