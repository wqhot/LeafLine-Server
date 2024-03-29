#include "include.h"  
#include "GaborFR.h"  
GaborFR::GaborFR()  
{  
    isInited = false;  
}  
void GaborFR::Init(cv::Size ksize, double sigma,double gamma, int ktype)
{  
    gaborRealKernels.clear();  
    gaborImagKernels.clear();  
    double mu[8]={0,1,2,3,4,5,6,7};  
    double nu[5]={0,1,2,3,4};  
    int i,j;  
    for(i=0;i<5;i++)  
    {  
        for(j=0;j<8;j++)  
        {  
            gaborRealKernels.push_back(getRealGaborKernel(ksize,sigma,mu[j]*CV_PI/8,nu[i],gamma,ktype));  
            gaborImagKernels.push_back(getImagGaborKernel(ksize,sigma,mu[j]*CV_PI/8,nu[i],gamma,ktype));  
        }  
    }  
    isInited = true;  
}  
cv::Mat GaborFR::getImagGaborKernel(cv::Size ksize, double sigma, double theta, double nu,double gamma, int ktype)
{  
    double  sigma_x     = sigma;  
    double  sigma_y     = sigma/gamma;  
    int     nstds       = 3;  
    double  kmax        = CV_PI/2;  
    double  f           = cv::sqrt(2.0);  
    int xmin, xmax, ymin, ymax;  
    double c = cos(theta), s = sin(theta);  
    if( ksize.width > 0 )  
    {  
        xmax = ksize.width/2;  
    }  
    else//这个和matlab中的结果一样，默认都是19 !  
    {  
        xmax = cvRound(std::max(fabs(nstds*sigma_x*c), fabs(nstds*sigma_y*s)));  
    }  
    if( ksize.height > 0 )  
    {  
        ymax = ksize.height/2;  
    }  
    else  
    {  
        ymax = cvRound(std::max(fabs(nstds*sigma_x*s), fabs(nstds*sigma_y*c)));  
    }  
    xmin = -xmax;  
    ymin = -ymax;  
    CV_Assert( ktype == CV_32F || ktype == CV_64F );  
    float*  pFloat;  
    double* pDouble;  
    cv::Mat kernel(ymax - ymin + 1, xmax - xmin + 1, ktype);  
    double k        =   kmax/pow(f,nu);  
    double scaleReal=   k*k/sigma_x/sigma_y;  
    for( int y = ymin; y <= ymax; y++ )  
    {  
        if( ktype == CV_32F )  
        {  
            pFloat = kernel.ptr<float>(ymax-y);  
        }  
        else  
        {  
            pDouble = kernel.ptr<double>(ymax-y);  
        }  
        for( int x = xmin; x <= xmax; x++ )  
        {  
            double xr = x*c + y*s;  
            double v = scaleReal*exp(-(x*x+y*y)*scaleReal/2);  
            double temp=sin(k*xr);  
            v   =  temp*v;  
            if( ktype == CV_32F )  
            {  
                pFloat[xmax - x]= (float)v;  
            }  
            else  
            {  
                pDouble[xmax - x] = v;  
            }  
        }  
    }  
    return kernel;  
}
cv::Mat GaborFR::getRealGaborKernel1(cv::Size ksize,double sigma,double theta,double nu,double gamma,int ktype)
{
	cv::Mat kernel;
	return kernel;
}
//sigma一般为2*pi  
//theta:方向；nu：尺度 gam
cv::Mat GaborFR::getRealGaborKernel(cv::Size ksize, double sigma, double theta,
    double nu,double gamma, int ktype)  
{  
    double  sigma_x     = sigma;  
    double  sigma_y     = sigma/gamma;  
    int     nstds       = 3;  
    double  kmax        = CV_PI/2;  
    double  f           = cv::sqrt(2.0);  
    int xmin, xmax, ymin, ymax;  
    double c = cos(theta), s = sin(theta);  
    if( ksize.width > 0 )  
    {  
        xmax = ksize.width/2;  
    }  
    else//这个和matlab中的结果一样，默认都是19 !  
    {  
        xmax = cvRound(std::max(fabs(nstds*sigma_x*c), fabs(nstds*sigma_y*s)));  
    }  
  
    if( ksize.height > 0 )  
        ymax = ksize.height/2;  
    else  
        ymax = cvRound(std::max(fabs(nstds*sigma_x*s), fabs(nstds*sigma_y*c)));  
    xmin = -xmax;  
    ymin = -ymax;  
    CV_Assert( ktype == CV_32F || ktype == CV_64F );  
    float*  pFloat;  
    double* pDouble;  
    cv::Mat kernel(ymax - ymin + 1, xmax - xmin + 1, ktype);  
    double k        =   kmax/pow(f,nu);  
    double exy      =   sigma_x*sigma_y/2;  
    double scaleReal=   k*k/sigma_x/sigma_y;  
    int    x,y;  
    for( y = ymin; y <= ymax; y++ )  
    {  
        if( ktype == CV_32F )  
        {  
            pFloat = kernel.ptr<float>(ymax-y);  
        }  
        else  
        {  
            pDouble = kernel.ptr<double>(ymax-y);  
        }  
        for( x = xmin; x <= xmax; x++ )  
        {  
            double xr = x*c + y*s;  
            double v = scaleReal*exp(-(x*x+y*y)*scaleReal/2);  
            double temp=cos(k*xr) - exp(-exy);  
            v   =   temp*v;  
            if( ktype == CV_32F )  
            {  
                pFloat[xmax - x]= (float)v;  
            }  
            else  
            {  
                pDouble[xmax - x] = v;  
            }  
        }  
    }  
    return kernel;  
}  
cv::Mat GaborFR::getMagnitude(cv::Mat &real,cv::Mat &imag)  
{  
    CV_Assert(real.type()==imag.type());  
    CV_Assert(real.size()==imag.size());  
    int ktype=real.type();  
    int row = real.rows,col = real.cols;  
    int i,j;  
    float*  pFloat,*pFloatR,*pFloatI;  
    double* pDouble,*pDoubleR,*pDoubleI;  
    cv::Mat     kernel(row, col, real.type());  
    for(i=0;i<row;i++)  
    {  
        if( ktype == CV_32FC1 )  
        {  
            pFloat = kernel.ptr<float>(i);  
            pFloatR= real.ptr<float>(i);  
            pFloatI= imag.ptr<float>(i);  
        }  
        else  
        {  
            pDouble = kernel.ptr<double>(i);  
            pDoubleR= real.ptr<double>(i);  
            pDoubleI= imag.ptr<double>(i);  
        }  
        for(j=0;j<col;j++)  
        {  
            if( ktype == CV_32FC1 )  
            {  
                pFloat[j]= sqrt(pFloatI[j]*pFloatI[j]+pFloatR[j]*pFloatR[j]);  
            }  
            else  
            {  
                pDouble[j] = sqrt(pDoubleI[j]*pDoubleI[j]+pDoubleR[j]*pDoubleR[j]);  
            }  
        }  
    }  
    return kernel;  
}  
cv::Mat GaborFR::getPhase(cv::Mat &real,cv::Mat &imag)  
{  
    CV_Assert(real.type()==imag.type());  
    CV_Assert(real.size()==imag.size());  
    int ktype=real.type();  
    int row = real.rows,col = real.cols;  
    int i,j;  
    float*  pFloat,*pFloatR,*pFloatI;  
    double* pDouble,*pDoubleR,*pDoubleI;  
    cv::Mat     kernel(row, col, real.type());  
    for(i=0;i<row;i++)  
    {  
        if( ktype == CV_32FC1 )  
        {  
            pFloat = kernel.ptr<float>(i);  
            pFloatR= real.ptr<float>(i);  
            pFloatI= imag.ptr<float>(i);  
        }  
        else  
        {  
            pDouble = kernel.ptr<double>(i);  
            pDoubleR= real.ptr<double>(i);  
            pDoubleI= imag.ptr<double>(i);  
        }  
        for(j=0;j<col;j++)  
        {  
            if( ktype == CV_32FC1 )  
            {  
//              if(pFloatI[j]/(pFloatR[j]+pFloatI[j]) > 0.99)  
//              {  
//                  pFloat[j]=CV_PI/2;  
//              }  
//              else  
//              {  
//                  pFloat[j] = atan(pFloatI[j]/pFloatR[j]);  
                pFloat[j] = asin(pFloatI[j]/sqrt(pFloatR[j]*pFloatR[j]+pFloatI[j]*pFloatI[j]));  
/*              }*/  
//              pFloat[j] = atan2(pFloatI[j],pFloatR[j]);  
            }//CV_32F  
            else  
            {  
                if(pDoubleI[j]/(pDoubleR[j]+pDoubleI[j]) > 0.99)  
                {  
                    pDouble[j]=CV_PI/2;  
                }  
                else  
                {  
                    pDouble[j] = atan(pDoubleI[j]/pDoubleR[j]);  
                }  
//              pDouble[j]=atan2(pDoubleI[j],pDoubleR[j]);  
            }//CV_64F  
        }  
    }  
    return kernel;  
}  
cv::Mat GaborFR::getFilterRealPart(cv::Mat& src, cv::Mat& real)
{  
    CV_Assert(real.type()==src.type());  
    cv::Mat dst;  
    cv::Mat kernel;  
    flip(real,kernel,-1);//中心镜面  
//  filter2D(src,dst,CV_32F,kernel,Point(-1,-1),0,BORDER_CONSTANT);  
	cv::filter2D(src,dst,CV_32F,kernel, cv::Point(-1,-1),0, cv::BORDER_REPLICATE);
    return dst;  
}  
cv::Mat GaborFR::getFilterImagPart(cv::Mat& src, cv::Mat& imag)
{  
    CV_Assert(imag.type()==src.type());  
    cv::Mat dst;  
    cv::Mat kernel;  
    flip(imag,kernel,-1);//中心镜面  
//  filter2D(src,dst,CV_32F,kernel,Point(-1,-1),0,BORDER_CONSTANT);  
	cv::filter2D(src,dst,CV_32F,kernel, cv::Point(-1,-1),0, cv::BORDER_REPLICATE);
    return dst;  
}  
void GaborFR::getFilterRealImagPart(cv::Mat& src, cv::Mat& real, cv::Mat& imag,cv::Mat &outReal,cv::Mat &outImag)
{  
    outReal=getFilterRealPart(src,real);  
    outImag=getFilterImagPart(src,imag);  
}  