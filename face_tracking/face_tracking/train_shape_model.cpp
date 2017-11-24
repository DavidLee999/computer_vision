//
//  train_shape_model.cpp
//  face_tracking
//
//  Created by 李芃桦 on 2017/11/24.
//  Copyright © 2017年 李芃桦. All rights reserved.
//

#include <opencv2/highgui/highgui.hpp>

#include <iostream>

#include "ft.hpp"

const char* usage =
"usage: ./train_shape_model annotation_file shape_model_file "
"[-f fraction_of_variation] [-k maximum_modes] [--mirror]";

bool sparse_help(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        string str = argv[i];
        if (str.length() == 2)
        {
            if (strcmp(str.c_str(), "-h") == 0)
                return true;
        }
        if (str.length() == 6)
        {
            if (strcmp(str.c_str(), "--help") == 0)
                return true;
        }
    }
    return false;
}

float parse_frac(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i)
    {
        string str = argv[i];
        if (str.length() != 2)
            continue;
        if (strcmp(str.c_str(), "-f") == 0)
        {
            if (argc > i + 1)
                return atof(argv[i + 1]);
        }
    }
    
    return 0.95;
}

float parse_kmax(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i)
    {
        string str = argv[i];
        if (str.length() != 2)
            continue;
        if (strcmp(str.c_str(), "-k"))
        {
            if (argc > i + 1)
                return atoi(argv[i + 1]);
        }
    }
    
    return 20;
}

bool parse_mirror(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i)
    {
        string str = argv[i];
        if (str.length() != 8)
            continue;
        if (strcmp(str.c_str(), "--mirror") == 0)
            return true;
    }
    
    return false;
}

int main(int argc, char** argv)
{
    // load data
    if (argc < 3)
    {
        cout << usage << endl;
        return 0;
    }
    
    float frac = parse_frac(argc, argv);
    int kmax = parse_kmax(argc, argv);
    bool mirror = parse_mirror(argc, argc);
    
    ft_data  data = load_ft<ft_data>(argv[1]);
    
}
