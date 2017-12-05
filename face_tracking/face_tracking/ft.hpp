//
//  ft.hpp
//  face_tracking
//
//  Created by 李芃桦 on 2017/11/24.
//  Copyright © 2017年 李芃桦. All rights reserved.
//

#ifndef ft_h
#define ft_h

#include "shape_model.hpp"
#include "ft_data.hpp"
#include "patch_model.cpp"

template <typename T>
T load_ft(const char* fname) {
    T x;
    FileStorage f(fname, FileStorage::READ);
    
    f["ft object"] >> x;
    
    f.release();
    
    return x;
}

template <typename T>
void save_ft(const char* fname, const T& x) {
    FileStorage f(fname, FileStorage::WRITE);
    
    f << "ft object" << x;
    
    f.release();
}

template <typename T>
void write(FileStorage& fs, const string&, const T& x) {
    x.write(fs);
}

template <typename T>
void read(const FileNode& node, T& x, const T& d) {
    if (node.empty())
        x = d;
    else
        x.read(node);
}

#endif /* ft_h */
