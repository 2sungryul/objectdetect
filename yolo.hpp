// title : yolo.hpp
// date : 2021.11.17 created, 2023.5.19 updated
// author : sungryul lee

#ifndef _YOLO_H_
#define _YOLO_H_

#include <iostream>
#include <queue>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cmath>
//#include <opencv2/cudacodec.hpp>
using namespace std;
using namespace cv;
using namespace cv::dnn;

Net yolo_init(const String& cfgfile, const String& weightfile, const String& classnames);
int yolo_inference(Net& net,cv::Size size, Mat& frame);

#endif //_YOLO_H_




