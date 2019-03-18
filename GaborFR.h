#pragma once  
#include "opencv2\opencv.hpp"  
#include <vector>  
using namespace std;  
class GaborFR  
{  
public:  
    GaborFR();  
    static cv::Mat  getImagGaborKernel(cv::Size ksize, double sigma, double theta,
                                    double nu,double gamma=1, int ktype= CV_32F);  
    static cv::Mat  getRealGaborKernel(cv::Size ksize, double sigma, double theta,
                                    double nu,double gamma=1, int ktype= CV_32F);  
    static cv::Mat  getPhase(cv::Mat &real, cv::Mat &imag);
    static cv::Mat  getMagnitude(cv::Mat &real, cv::Mat &imag);
    static void getFilterRealImagPart(cv::Mat& src, cv::Mat& real, cv::Mat& imag, cv::Mat &outReal, cv::Mat &outImag);
    static cv::Mat  getFilterRealPart(cv::Mat& src, cv::Mat& real);
    static cv::Mat  getFilterImagPart(cv::Mat& src, cv::Mat& imag);
	static cv::Mat getRealGaborKernel1(cv::Size ksize,double sigma,double theta,double nu,double gamma,int ktype=CV_32F);
    void        Init(cv::Size ksize= cv::Size(19,19), double sigma=2*CV_PI,
                    double gamma=1, int ktype=CV_32FC1);  
private:  
    vector<cv::Mat> gaborRealKernels;
    vector<cv::Mat> gaborImagKernels;
    bool isInited;  
};  

