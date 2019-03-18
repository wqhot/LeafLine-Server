#include "gp.h"
#include <math.h>
#define GLCM_DIS 1  //灰度共生矩阵的统计距离  
#define GLCM_CLASS 8 //计算灰度共生矩阵的图像灰度值等级化  
#define GLCM_ANGLE_HORIZATION 0  //水平  
#define GLCM_ANGLE_VERTICAL   1  //垂直  
#define GLCM_ANGLE_DIGONAL    2  //对角 
void neiQie(vector<vector<cv::Point> > contours,double* neiqie,int big);
int cornerShiTomasi_demo(cv::Mat  src_gray,cv::Mat &draw) ;
void m_calHu(cv::Mat F,double *hu);

cv::Mat shapeFeatureCal(cv::Mat bw3,double* P) 
{
	cv::Mat bw1 = bw3.clone();
	cv::Mat draw(bw3.size(),CV_8UC3, cv::Scalar(0,0,0));
	cvtColor(bw3,draw, cv::COLOR_GRAY2RGB);
	//cv::Mat draw=bw3.clone();
	cv::Mat conrnerImg = bw3.clone();
	
	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;
	findContours(bw1 ,contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
	
	int big;
	int bigsize=0;
	for(int i=0;i<contours.size();i++)
	{
		if(bigsize<contours[i].size())
		{
			bigsize=contours[i].size();
			big=i;
		}
	}
	cv::Rect r = boundingRect(contours[big]);
	rectangle(draw, cv::Point(r.x,r.y), cv::Point(r.x+r.width,r.y+r.height), cv::Scalar(255,255,0),2);//yellow
	
	double A = (double) r.height / (double) r.width;//狭长度
	if (A < 1) {
		A = 1 / A;
	}
	double B = contourArea(contours[big]) / (double) r.height / (double) r.width;//矩形度
	float bigRadius[1] ;
	cv::Point2f bigCenter;
	//vector<vector<Point> > contours_poly( contours.size() );  
	//approxPolyDP( Mat(contours[big]), contours_poly[0], 3, true );  
	minEnclosingCircle(contours[big], bigCenter, bigRadius[0]);
	circle(draw,bigCenter,bigRadius[0], cv::Scalar(0,128,255),2);//blue
	double C = 4 * 3.1415927 * contourArea(contours[big]) / (double) bigRadius[0] / (double) bigRadius[0];//球状性
	double*neiqie=(double *)malloc(3*sizeof(double));
	neiQie(contours,neiqie,big);
	double D = (double) bigRadius[0] / (double) *neiqie;//圆形度
	cv::RotatedRect minEllipse;
	minEllipse = fitEllipse(contours[big]);
	ellipse(draw,minEllipse, cv::Scalar(0,128,0),2);//green
	double E = (double) minEllipse.size.height / (double) minEllipse.size.width;//偏心率
	vector<cv::Point> hull;
	convexHull(contours[big], hull);
	//cv::Mat hierarchy1 = new Mat();
	//MatOfPoint hullpoints = hull2Points(hull, contours.get(0));
	//drawContours(draw,hull,big,Scalar(255));
	vector<cv::Point>::const_iterator it= hull.begin();
	while (it!=(hull.end()-1)) {
		line(draw,*it,*(it+1), cv::Scalar(255,0,0),2);//red
		++it;
	}
	//MatOfPoint2f newMat1 = new MatOfPoint2f(hullpoints.toArray());
	double F = arcLength(hull, true) / (double) bigRadius[0] / (double) bigRadius[0];//周长直径比
	double G = arcLength(hull, true) / ((double) r.height + (double) r.width);//周长长宽比
	
	int Ha = cornerShiTomasi_demo(conrnerImg,draw);
	//cout<<Ha;
	//cout<<" "<<arcLength(hull, true);
	double H=((double) Ha / arcLength(hull, true));
	*(P+0)=A;*(P+1)=B;*(P+2)=C;*(P+3)=D;
	*(P+4)=E;*(P+5)=F;*(P+6)=G;*(P+7)=H;
	double hu[7];
	m_calHu(bw3.clone(),hu);
	/*Moments mu = moments( bw3.clone(), true );	
	HuMoments(mu,hu);*/
	for(int i=0;i<7;i++)
	{
		*(P+i+8)=log10(abs(hu[i]));
		//cout<<*(P+i+7)<<" ";
	}
	//waitKey(0);
	return draw;
}

void m_calHu(cv::Mat F,double*hu)
{
	int n =F.cols;
        int m = F.rows;
		//imshow("s",F);
		//waitKey(0);
        int x, y, p, q;
       // double hu[7];
        double mm [4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                mm[i][j] = 0;
            }
        }
        //图像的各阶矩
        for (y = 0; y < m; y++) {
            for (x = 0; x < n; x++) {
                for (q = 0; q < 4; q++) {
                    for (p = 0; p < 4; p++) {
                        uchar temp = F.at<uchar>(y, x);
                        mm[q][p] = mm[q][p] + pow((double)x, p) * pow((double)y, q) * temp;
                    }
                }
            }
        }
        double mean_x = mm[1][0] / mm[0][0];
        double mean_y = mm[0][1] / mm[0][0];
        //三阶中心矩
        double u00 = mm[0][0];
        double u11 = mm[1][1] - mean_y * mm[1][0];

        double u20 = mm[2][0] - mean_x * mm[1][0];
        double u02 = mm[0][2] - mean_y * mm[0][1];
        double u30 = mm[3][0] - 3 * mean_x * mm[2][0] + 2 * std::pow(mean_x, 2) * mm[1][0];
        double u03 = mm[0][3] - 3 * mean_y * mm[0][2] + 2 * std::pow(mean_y, 2) * mm[0][1];
        double u21 = mm[2][1] - 2 * mean_x * mm[1][1] - mean_y * mm[2][0] + 2 * std::pow(mean_x, 2) * mm[0][1];
        double u12 = mm[1][2] - 2 * mean_y * mm[1][1] - mean_x * mm[0][2] + 2 * std::pow(mean_y, 2) * mm[1][0];
        //归一化中心矩
        double n20 = u20 / std::pow(u00, 2);
        double n02 = u02 / std::pow(u00, 2);
        double n11 = u11 / std::pow(u00, 2);
        double n30 = u30 / std::pow(u00, 2.5);
        double n03 = u03 / std::pow(u00, 2.5);
        double n12 = u12 / std::pow(u00, 2.5);
        double n21 = u21 / std::pow(u00, 2.5);
        //7个不变矩
        *(hu+0)= n20 + n02;
        *(hu+1)= std::pow((n20 - n02), 2) + 4 * std::pow(n11, 2);
        *(hu+2) = std::pow((n30 - 3 * n12), 2) + std::pow((3 * n21 - n03), 2);
        *(hu+3) = std::pow((n30 + n12), 2) + std::pow((n21 + n03), 2);
        *(hu+4) = (n30 - 3 * n12) * (n30 + n12) * (std::pow((n30 + n12), 2) - 3 * std::pow((n21 + n03), 2)) + (3 * n21 - n03) * (n21 + n03) * (3 * std::pow((n30 + n12), 2) - std::pow((n21 + n03), 2));
        *(hu+5) = (n20 - n02) * (std::pow((n30 + n12), 2) - std::pow((n21 + n03), 2)) + 4 * n11 * (n30 + n12) * (n21 + n03);
        *(hu+6) = (3 * n21 - n03) * (n30 + n12) * (std::pow((n30 + n12), 2) - 3 * std::pow((n21 + n03), 2)) + (3 * n12 - n30) * (n21 + n03) * (3 * std::pow((n30 + n12), 2) - std::pow((n21 + n03), 2));

        //return hu; 
}


void neiQie(vector<vector<cv::Point> > contours,double* neiqie,int big) {
        cv::Rect r = boundingRect(contours[big]);
        int dx = r.x + r.width;
        int dy = r.y + r.height;
        int rx = 0, ry = 0;
        double R = 2;
        for (int x = r.x; x < dx; x += 5) {
            for (int y = r.y; y < dy; y += 5) {
                //MatOfPoint2f newcv::Mat = new MatOfPoint2f(contours.toArray());
                double d = pointPolygonTest(contours[big], cv::Point(x, y), true);
                if (d > 0 && R < d) {
                    R = d;
                    rx = x;
                    ry = y;
                }
            }
        }
        //second
        dx = rx + 5;
        dy = ry + 5;
        for (int x = rx; x < dx; x += 1)
            for (int y = ry; y < dy; y += 1) {
               
                double d = pointPolygonTest(contours[big], cv::Point(x, y), true);
                if (d > 0 && R < d) {
                    rx = x;
                    ry = y;
                    R = d;
                }
            }

      
        *(neiqie) = R;
        *(neiqie+1)  = rx;
        *(neiqie+2)  = ry;

    }
int cornerShiTomasi_demo(cv::Mat  src_gray,cv::Mat &draw)  
{  
  int maxCorners=100;
  /// Parameters for Shi-Tomasi algorithm  
  vector<cv::Point2f> corners;
  double qualityLevel = 0.01;  
  double minDistance = 10;  
  int blockSize = 3;  
  bool useHarrisDetector = false;  
  double k = 0.04;  
  /// Copy the source image  
  cv::Mat cormat=src_gray.clone();  
  /// Apply corner detection :Determines strong corners on an image.  
  goodFeaturesToTrack( cormat,   
               corners,  
               maxCorners,  
               qualityLevel,  
               minDistance,  
			   cv::Mat(),
               blockSize,  
               useHarrisDetector,  
               k );  
  for( int i = 0; i < corners.size(); i++ ){   
      
      circle( draw, corners[i], 4, cv::Scalar(255) );
  }
  return corners.size();
}  
void m_calHu(cv::Mat F)
{
	int n = F.cols;
        int m = F.rows;
        int x, y, p, q;
        double hu[7];
        double mm[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                mm[i][j] = 0;
            }
        }
        //图像的各阶矩
        for (y = 0; y < m; y++) {
            for (x = 0; x < n; x++) {
                for (q = 0; q < 4; q++) {
                    for (p = 0; p < 4; p++) {
						int temp = F.at<uchar>(y,x);
                        mm[q][p] = mm[q][p] + pow((double)x, p) * pow((double)y, q) * temp;
                    }
                }
            }
        }
        double mean_x = mm[1][0] / mm[0][0];
        double mean_y = mm[0][1] / mm[0][0];
        //三阶中心矩
        double u00 = mm[0][0];
        double u11 = mm[1][1] - mean_y * mm[1][0];

        double u20 = mm[2][0] - mean_x * mm[1][0];
        double u02 = mm[0][2] - mean_y * mm[0][1];
        double u30 = mm[3][0] - 3 * mean_x * mm[2][0] + 2 * pow(mean_x, 2) * mm[1][0];
        double u03 = mm[0][3] - 3 * mean_y * mm[0][2] + 2 * pow(mean_y, 2) * mm[0][1];
        double u21 = mm[2][1] - 2 * mean_x * mm[1][1] - mean_y * mm[2][0] + 2 * pow(mean_x, 2) * mm[0][1];
        double u12 = mm[1][2] - 2 * mean_y * mm[1][1] - mean_x * mm[0][2] + 2 * pow(mean_y, 2) * mm[1][0];
        //归一化中心矩
        double n20 = u20 / pow(u00, 2);
        double n02 = u02 / pow(u00, 2);
        double n11 = u11 / pow(u00, 2);
        double n30 = u30 / pow(u00, 2.5);
        double n03 = u03 / pow(u00, 2.5);
        double n12 = u12 / pow(u00, 2.5);
        double n21 = u21 / pow(u00, 2.5);
        //7个不变矩
        hu[0] = n20 + n02;
        hu[1] = pow((n20 - n02), 2) + 4 * pow(n11, 2);
        hu[2] = pow((n30 - 3 * n12), 2) + pow((3 * n21 - n03), 2);
        hu[3] = pow((n30 + n12), 2) + pow((n21 + n03), 2);
        hu[4] = (n30 - 3 * n12) * (n30 + n12) * (pow((n30 + n12), 2) - 3 * pow((n21 + n03), 2)) + (3 * n21 - n03) * (n21 + n03) * (3 * pow((n30 + n12), 2) - pow((n21 + n03), 2));
        hu[5] = (n20 - n02) * (pow((n30 + n12), 2) - pow((n21 + n03), 2)) + 4 * n11 * (n30 + n12) * (n21 + n03);
        hu[6] = (3 * n21 - n03) * (n30 + n12) * (pow((n30 + n12), 2) - 3 * pow((n21 + n03), 2)) + (3 * n12 - n30) * (n21 + n03) * (3 * pow((n30 + n12), 2) -pow((n21 + n03), 2));

       
    }

int calGLCM(cv::Mat bWavelet,int angleDirection,double* featureVector)  
{  
    int i,j;  
    int width,height;  
  
     
  
    width = bWavelet.cols;  
    height = bWavelet.rows;  
  
    int * glcm = new int[GLCM_CLASS * GLCM_CLASS];  
	double * normglcm = new double[GLCM_CLASS * GLCM_CLASS]; 
    int * histImage = new int[width * height];  
  
    if(NULL == glcm || NULL == histImage)  
        return 2;  
  
    //灰度等级化---分GLCM_CLASS个等级  
	uchar *data =(uchar*) bWavelet.data;  
    for(i = 0;i < height;i++){  
        for(j = 0;j < width;j++){  
            histImage[i * width + j] = (int)(data[bWavelet.cols * i + j] * GLCM_CLASS / 256);  
        }  
    }  
  
    //初始化共生矩阵  
    for (i = 0;i < GLCM_CLASS;i++)  
        for (j = 0;j < GLCM_CLASS;j++)  
            glcm[i * GLCM_CLASS + j] = 0;  
  
    //计算灰度共生矩阵  
    int w,k,l;  
    //水平方向  
    if(angleDirection == GLCM_ANGLE_HORIZATION)  
    {  
        for (i = 0;i < height;i++)  
        {  
            for (j = 0;j < width;j++)  
            {  
                l = histImage[i * width + j];  
                if(j + GLCM_DIS >= 0 && j + GLCM_DIS < width)  
                {  
                    k = histImage[i * width + j + GLCM_DIS];  
                    glcm[l * GLCM_CLASS + k]++;  
                }  
                if(j - GLCM_DIS >= 0 && j - GLCM_DIS < width)  
                {  
                    k = histImage[i * width + j - GLCM_DIS];  
                    glcm[l * GLCM_CLASS + k]++;  
                }  
            }  
        }  
    }  
    //垂直方向  
    else if(angleDirection == GLCM_ANGLE_VERTICAL)  
    {  
        for (i = 0;i < height;i++)  
        {  
            for (j = 0;j < width;j++)  
            {  
                l = histImage[i * width + j];  
                if(i + GLCM_DIS >= 0 && i + GLCM_DIS < height)   
                {  
                    k = histImage[(i + GLCM_DIS) * width + j];  
                    glcm[l * GLCM_CLASS + k]++;  
                }  
                if(i - GLCM_DIS >= 0 && i - GLCM_DIS < height)   
                {  
                    k = histImage[(i - GLCM_DIS) * width + j];  
                    glcm[l * GLCM_CLASS + k]++;  
                }  
            }  
        }  
    }  
    //对角方向  
    else if(angleDirection == GLCM_ANGLE_DIGONAL)  
    {  
        for (i = 0;i < height;i++)  
        {  
            for (j = 0;j < width;j++)  
            {  
                l = histImage[i * width + j];  
  
                if(j + GLCM_DIS >= 0 && j + GLCM_DIS < width && i + GLCM_DIS >= 0 && i + GLCM_DIS < height)  
                {  
                    k = histImage[(i + GLCM_DIS) * width + j + GLCM_DIS];  
                    glcm[l * GLCM_CLASS + k]++;  
                }  
                if(j - GLCM_DIS >= 0 && j - GLCM_DIS < width && i - GLCM_DIS >= 0 && i - GLCM_DIS < height)  
                {  
                    k = histImage[(i - GLCM_DIS) * width + j - GLCM_DIS];  
                    glcm[l * GLCM_CLASS + k]++;  
                }  
            }  
        }  
    }  
	/*for(int i=0;i<GLCM_CLASS;i++)
	{
		for(int j=0;j<GLCM_CLASS;j++)
			cout<<glcm[i* GLCM_CLASS+j]<<" ";
		cout<<"\n";
	}
	*/
	 int sum = 0;
        for (i = 0; i < GLCM_CLASS; i++) {
            for (j = 0; j < GLCM_CLASS; j++) {
                sum = sum + glcm[i * GLCM_CLASS + j];
            }
        }

    for (i = 0; i < GLCM_CLASS; i++) {

            for (j = 0; j < GLCM_CLASS; j++) {
                normglcm[i * GLCM_CLASS + j] = (double) (glcm[i * GLCM_CLASS + j]) / sum;

            }

        }

	
    //计算特征值  
    double entropy = 0,energy = 0,contrast = 0,homogenity = 0;  
    for (i = 0;i < GLCM_CLASS;i++)  
    {  
        for (j = 0;j < GLCM_CLASS;j++)  
        {  
            //熵  
            if(normglcm[i * GLCM_CLASS + j] > 0)  
                entropy -= normglcm[i * GLCM_CLASS + j] * log10(double(normglcm[i * GLCM_CLASS + j]));  
            //能量  
            energy += normglcm[i * GLCM_CLASS + j] * normglcm[i * GLCM_CLASS + j];  
            //对比度  
            contrast += (i - j) * (i - j) * normglcm[i * GLCM_CLASS + j];  
            //一致性  
            homogenity += 1.0 / (1 + (i - j) * (i - j)) * normglcm[i * GLCM_CLASS + j];  
        }  
    }  
    //返回特征值  
    i = 0;  
    featureVector[i++] = entropy;  
    featureVector[i++] = energy;  
    featureVector[i++] = contrast;  
    featureVector[i++] = homogenity;  
	

    delete[] glcm;  
    delete[] histImage;  
    return 0;  
} 
