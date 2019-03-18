#include "proc.h"
#define MATSIZE 400
cv::Mat rotateMat(double angle,cv::Mat inputImg);
vector<cv::Mat> preCal(string filename)
{
	//const char* filename="84.jpg";
	cv::Mat srcImg= cv::imread(filename,CV_LOAD_IMAGE_COLOR);

	cv::Mat grayImg;
	cv::Mat bwImg;
	cvtColor(srcImg,grayImg, cv::COLOR_RGB2GRAY);
	
	threshold(grayImg,bwImg,0,255, cv::THRESH_BINARY| cv::THRESH_OTSU);
	//cout<<bwImg;
	cv::Mat element =getStructuringElement(cv::MORPH_RECT, cv::Size(11,11), cv::Point(5, 5 ));
	//morphologyEx(bwImg,bwImg,MORPH_OPEN,element);
	//morphologyEx(bwImg,bwImg,MORPH_CLOSE,element);
	bitwise_not(bwImg,bwImg);
	//morphologyEx(bwImg,bwImg,MORPH_OPEN,element);
	
	
	bwImg=m_BiggestArea(bwImg);
	// imwrite("test0.jpg",bwImg);

	threshold(bwImg,bwImg,0.5,1, cv::THRESH_BINARY);
	grayImg=grayImg.mul(bwImg);
	threshold(bwImg,bwImg,0.5,255, cv::THRESH_BINARY);
	vector<cv::Mat> img;
	img=minBox(bwImg,grayImg);
	//imshow("a",bwImg);
	
	return img;
}

vector<cv::Mat> minBox(cv::InputArray _src, cv::InputArray _gray)
{
	cv::Mat src1=_src.getMat();
	cv::Mat src;
	src1.copyTo(src);
	cv::Mat output=_gray.getMat();
	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;
	//imwrite("test1.jpg",src);
	
	findContours(src ,contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
	//imshow("1",src1);
	
	vector<cv::RotatedRect> tuoyuan( contours.size());
	// vector<Rect> boundRect( contours.size() );
	  vector<vector<cv::Point> > contours_poly( contours.size() );
	  tuoyuan[0]=fitEllipse(contours[0]);
 
	cv::Mat src2=rotateMat( tuoyuan[0].angle,src1);
	cv::Mat output2=rotateMat( tuoyuan[0].angle,output);
	cv::Mat output1=src2.clone();

	findContours(src2 ,contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
	vector<cv::Rect> boundRect( contours.size() );


  approxPolyDP(cv::Mat(contours[0]), contours_poly[0], 3, true );
  boundRect[0] = boundingRect(cv::Mat(contours_poly[0]) );
  output2=output2(boundRect[0]);
  output1=output1(boundRect[0]);
  int height=output2.rows;
  int width=output2.cols;
  cv::Mat out_new(MATSIZE,MATSIZE,CV_8UC1, cv::Scalar(0));
  cv::Mat out_new1(MATSIZE,MATSIZE,CV_8UC1, cv::Scalar(0));
 int flag=1;
 if (height<width)
  {
	  int temp=height;
	  height=width;
	  width=temp;
	  output2=output2.t();
	  output1=output1.t();
	  flag=0;
	 	 
  }

	  double scale=(double)MATSIZE/height;
	  resize(output2,output2, cv::Size(width*scale,height*scale));
	  resize(output1,output1, cv::Size(width*scale,height*scale));
	  // cout<<output2.rows<<","<<output2.cols<<"\n";
	 
//	  cout<<out_new(Rect(0,floor((400-height*scale)/2),output2.cols,output2.rows)).rows<<","<<out_new(Rect(0,floor((400-height*scale)/2),output2.cols,output2.rows)).cols;
	output1.copyTo( out_new1(cv::Rect(floor((MATSIZE-width*scale)/2),0,output2.cols,output2.rows)));
	  output2.copyTo( out_new(cv::Rect(floor((MATSIZE-width*scale)/2),0,output2.cols,output2.rows)));
	if(flag==0)
	{
		out_new=out_new.t();
		out_new1=out_new1.t();
	}
	  //out_new(Rect(floor((400-width*scale)/2),0,output.cols,output.rows))=output;

  //imshow("source", out_new);
  vector<cv::Mat> ret;
  ret.push_back(out_new1);
  ret.push_back(out_new);
  return ret;

}
cv::Mat rotateMat(double angle,cv::Mat inputImg)
{
	  float radian = (float) (angle /180.0 * CV_PI);
	  cv::Mat tempImg;
        //填充图像使其符合旋转要求
        int uniSize =(int) ( max(inputImg.cols, inputImg.rows)* 1.414 );
        int dx = (int) (uniSize - inputImg.cols)/2;
        int dy = (int) (uniSize - inputImg.rows)/2;

        copyMakeBorder(inputImg, tempImg, dy, dy, dx, dx, cv::BORDER_CONSTANT);


        //旋轉中心
		cv::Point2f center( (float)(tempImg.cols/2) , (float) (tempImg.rows/2));
        cv::Mat affine_matrix = getRotationMatrix2D( center, angle, 1.0 );

        //旋轉
        warpAffine(tempImg, tempImg, affine_matrix, tempImg.size());


        //旋轉后的圖像大小
        float sinVal = fabs(sin(radian));
        float cosVal = fabs(cos(radian));

        
		cv::Size targetSize( (int)(inputImg.cols * cosVal + inputImg.rows * sinVal),
                (int)(inputImg.cols * sinVal + inputImg.rows * cosVal) );

        //剪掉四周边框
        int x = (tempImg.cols - targetSize.width) / 2;
        int y = (tempImg.rows - targetSize.height) / 2;

		cv::Rect rect(x, y, targetSize.width, targetSize.height);
        tempImg = cv::Mat(tempImg, rect);
		return tempImg;
}
cv::Mat m_BiggestArea(cv::InputArray  _src_mat) {
	
	 cv::Mat src_mat;
	 _src_mat.getMat().copyTo(src_mat);
	 int w,h;  
	
     int color = 254;
	//imwrite("test.jpg",src_mat);
	/*ofstream f("test.txt");
	f<<src_mat;
	f.close();*/
	 cv::Mat mask(src_mat.rows + 2, src_mat.cols + 2, CV_8UC1, cv::Scalar::all(0));
	 for (w = 0; w < src_mat.rows; w++) {
            for (h = 0; h < src_mat.cols; h++) {
                if (color > 0) {
                    
					if (src_mat.at<uchar>(w,h) == 255) {
						//cout<<"row="<<w<<",col="<<h<<":"<<(uchar)src_mat.at<uchar>(w,h)<<"\n" ;
						cv::Scalar scalar(color);
						cv::Point point(h,w) ;//(w, h);
						cv::floodFill(src_mat, mask, point, scalar);
                        color--;
                    }
                }
            }
        }
	 int colorsum[256];
        for (int i = 0; i < 256; i++) {
            colorsum[i] = 0;
        }
        for (w = 0; w < src_mat.rows; w++) {
            for (h = 0; h < src_mat.cols; h++) {
                
                if (src_mat.at<uchar>(w,h) > 0) {
                    colorsum[ src_mat.at<uchar>(w,h)]++;
                }
            }
        }
		 int max_color_sum = 0;
        int max_color = 0;
        for (int i = 0; i < 256; i++) {
            if (colorsum[i] > max_color_sum) {
                max_color_sum = colorsum[i];
                max_color = i;
            }
        }
		for (w = 0; w < src_mat.rows; w++) {
            for (h = 0; h < src_mat.cols; h++) {
                
                if (src_mat.at<uchar>(w,h) == max_color) {
                    src_mat.at<uchar>(w,h)=255;
                    //colorsum[(int)temp[0]]++;
                } else {
                   src_mat.at<uchar>(w,h)=0;
                }
            }
        }
		return src_mat;
}