#ifndef  __LBP_H_
#define  __LBP_H_
#include "include.h"
//cv::MatND hisggg(cv::Mat src);

cv::Mat lbp_hist(cv::Mat grayImg,double*hist,int grid_x, int grid_y,int nei);

#endif