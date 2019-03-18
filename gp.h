#ifndef  __GP_H_
#define  __GP_H_
#include "include.h"
cv::Mat shapeFeatureCal(cv::Mat bw3,double* P);
int calGLCM(cv::Mat bWavelet,int angleDirection,double* featureVector) ;
#endif