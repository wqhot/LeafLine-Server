#ifndef  __PROC_H_
#define  __PROC_H_
#include "include.h"
vector<cv::Mat> preCal(string filename);
vector<cv::Mat> minBox(cv::InputArray _src, cv::InputArray _gray);
cv::Mat m_BiggestArea(cv::InputArray  _src_mat);
#endif