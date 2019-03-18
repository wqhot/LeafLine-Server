#include "include.h"
#include "lbp.h"
#include "GaborFR.h"  
#include "proc.h"
#include "gp.h"
#include "datapre.h"
#include "datasvm.h"
#include "pca.h"

//int BLOCKNUM =7;
//#define BLOCKNUM_GABOR 4
int  CHIDU=0;

int FANGXIANG=0;


int BLOCKNUM_GABOR=0;
int BLOCKNUM =0;
void lbp_cal_all(string line,cv::Mat grayImg,cv::Mat bwImg);
cv::Mat gabor_cal_all(double*gabor,cv::Mat grayImg);
int h=0;
int oldclass=0;
int writeid=0;
ofstream f;
ofstream f1;
ofstream f2;
//ofstream x1;
int cal_flag=0;
int sel_flag=0;
int main()
{	
	system("cmd /c dir .\\photo\\*.jpg /a-d /b /s >allfiles.txt");
	settingdata settingData=ReadSetting();
	FANGXIANG=settingData.gaborDur;
	CHIDU=settingData.gaborF+CHIDU1;
	BLOCKNUM_GABOR=settingData.gaborBlockNum;
	BLOCKNUM=settingData.lbpBlockNum;
	ifstream in("allfiles.txt");
	sel_flag = 1;
	cal_flag = 7;
	BLOCKNUM = 7;
	BLOCKNUM_GABOR = 4;
	CHIDU = 5;
	FANGXIANG = 8;
	//cout<<"1————0"<<endl;
	//cout<<"^        ^"<<endl;
	//cout<<"|        |"<<endl;
	//cout<<"训练    计算"<<endl;
	//cin>>sel_flag;
	//if((sel_flag&0x01)==1)
	//{
	//	cout<<"2————1————0"<<endl;
	//	cout<<"^        ^        ^"<<endl;
	//	cout<<"|        |        |"<<endl;
	//	cout<<"20维    gabor    lbp"<<endl;
	//	cout<<"输入计算模式：";
	//	cin>>cal_flag;
	//}
	//else
	//{
	//	cal_flag=settingData.cal_flag;
	//}
	//if((cal_flag&0x01)==1&&(sel_flag&0x01)==1)
	//{
	//	f.open("lbp.txt");
	//	cout<<"输入LBP分块数量：";
	//	cin>>BLOCKNUM;
	//	//x1.open("lbp.csv");
	//	while(BLOCKNUM<=0)
	//	{
	//		cout<<"输入LBP分块数量：";
	//		cin>>BLOCKNUM;
	//	}
	//}
	//if((cal_flag&0x04)==4&&(sel_flag&0x01)==1)
	//{
	//	f2.open("study.txt");
	//}
	//
	//if((cal_flag&0x02)==2&&(sel_flag&0x01)==1)
	//{
	//	f1.open("gabor.txt");
	//	cout<<"输入Gabor分块数量：";
	//	cin>>BLOCKNUM_GABOR;
	//	while(BLOCKNUM_GABOR<=0)
	//	{
	//		cout<<"输入Gabor分块数量：";
	//		cin>>BLOCKNUM_GABOR;
	//	}
	//	cout<<"输入最大Gabor频率（尺度）：";
	//	cin>>CHIDU;
	//	while(CHIDU<=0)
	//	{
	//		cout<<"输入最大Gabor频率（尺度）：";
	//		cin>>CHIDU;
	//	}
	//	cout<<"输入Gabor方向个数：";
	//	cin>>FANGXIANG;
	//	while(FANGXIANG<=0)
	//	{
	//		cout<<"输入Gabor方向个数：";
	//		cin>>FANGXIANG;
	//	}
	//}
	SaveSetting(BLOCKNUM,BLOCKNUM_GABOR,CHIDU-CHIDU1,FANGXIANG,cal_flag);
	string filename;  
    string line; 
	cout << "Begin" << endl;
	if((sel_flag&0x01)==1)
	{
		if(in) // 有该文件  
		{  
			while (getline (in, line)) // line中不包括每行的换行符  
			{   
				cout << line;
				vector<cv::Mat> image;
				image=preCal(line);

				cv::Mat grayImg=image.at(1);
				cv::Mat bwImg=image.at(0);
			
			
				
				lbp_cal_all(line,grayImg,bwImg);
			}
			
		}  
	}
	
	//x1.close();
	f.close();
	f1.close();
	f2.close();
	if((sel_flag&0x02)==2)
	{
		//cout<<BLOCKNUM<<BLOCKNUM_GABOR<<CHIDU-CHIDU1<<FANGXIANG<<cal_flag<<endl;
		DataLoadAll(BLOCKNUM,BLOCKNUM_GABOR,CHIDU-CHIDU1,FANGXIANG,cal_flag);
		exeSVMTrain();
		//readdata("data2svm.txt");
		//readdata("data2svm.txt");
	}
		
}
cv::Mat gabor_cal_all(double*gabor,cv::Mat grayImg)
{	
	int iSize=11;
	//imshow("a", grayImg);
	//cv::waitKey(0);
	//normalize(grayImg,grayImg,1,0, cv::NORM_MINMAX,CV_32F);
	//cv::normalize()
	grayImg.convertTo(grayImg, CV_32F, 1.0/255.0);
	//imshow("a", grayImg);
	//cv::waitKey(0);
	cv::Mat showM,showMM;
	cv::Mat M,MatTemp1,MatTemp2;
	cv::Mat line;
	for(int i=0;i<FANGXIANG;i++)//方向
    {  
		showM.release();  
		for(int j=CHIDU1;j<CHIDU;j++)//尺度
		{
			cv::Mat M1= GaborFR::getRealGaborKernel(cv::Size(iSize,iSize),2*CV_PI,i*CV_PI/FANGXIANG+CV_PI/2, j,1);
            cv::Mat M2 = GaborFR::getImagGaborKernel(cv::Size(iSize,iSize),2*CV_PI,i*CV_PI/FANGXIANG+CV_PI/2, j,1);
			cv::Mat outR,outI;
			GaborFR::getFilterRealImagPart(grayImg,M1,M2,outR,outI); 
			cv::Mat out;
			M=GaborFR::getMagnitude(outR,outI);
			//imshow("a", M);
			//cv::waitKey(0);
			M.convertTo(M, CV_8U, 255.0);
			//normalize(M,M,0,255,CV_MINMAX,CV_8U);
			//imshow("a", M);
			//cv::waitKey(0);
			int height=M.rows/BLOCKNUM_GABOR;
			int width=M.cols/BLOCKNUM_GABOR;

			 for(int ii = 0; ii < BLOCKNUM_GABOR; ii++) 
			{
				for(int jj = 0; jj < BLOCKNUM_GABOR; jj++)
				 {
					 // 获取指定区域
					cv::Mat src_cell = cv::Mat(M, cv::Range(ii*height,(ii+1)*height), cv::Range(jj*width,(jj+1)*width));
					// 计算指定区域的直方图
					
					cv::Scalar m=mean(src_cell);
					double mm=m.val[0];
					cv::Scalar stddev;
					meanStdDev(src_cell,m,stddev);
					double ss=stddev.val[0];
					*(gabor+i*(CHIDU-CHIDU1)*BLOCKNUM_GABOR*BLOCKNUM_GABOR*2+j*BLOCKNUM_GABOR*BLOCKNUM_GABOR*2+ii*BLOCKNUM_GABOR*2+jj*2)=mm;
					*(gabor+i*(CHIDU-CHIDU1)*BLOCKNUM_GABOR*BLOCKNUM_GABOR*2+j*BLOCKNUM_GABOR*BLOCKNUM_GABOR*2+ii*BLOCKNUM_GABOR*2+jj*2+1)=ss;
					//cout<<i*CHIDU*BLOCKNUM_GABOR*BLOCKNUM_GABOR*2+j*BLOCKNUM_GABOR*BLOCKNUM_GABOR*2+ii*BLOCKNUM_GABOR*2+jj*2<<",";
				}
			}
			/*log(outR.mul(outR)+outI.mul(outI),out);
			out=out.mul(0.5);
			exp(out,out);*/
			resize(M1,M1, cv::Size(100,100));
			normalize(M1,M1,0,255,CV_MINMAX,CV_8U);
			//imshow("M",M1);
			/*imshow("S",M);
			
			waitKey(0);*/

			normalize(M,M,0,255,CV_MINMAX,CV_8U);
			////normalize(M1,M1,0,255,CV_MINMAX,CV_8U);
			////imshow("M",M1);
			resize(M,M, cv::Size(150,150));
			//
			showM.push_back(M); 
			line= cv::Mat::ones(4,M.cols,M.type())*255;
            showM.push_back(line); 
		}
		showM=showM.t(); 
		line= cv::Mat::ones(4,showM.cols,showM.type())*255;
        showMM.push_back(showM);  
        showMM.push_back(line);   
	}
	//return M;
	//free(gabor);
	showMM=showMM.t();
	return showMM;
	/*imshow("saveMM",showMM);
	waitKey(0);*/
}

void lbp_cal_all(string line,cv::Mat grayImg,cv::Mat bwImg)
{
	cv::Mat gaborImg;
	cv::Mat lbpImg;
	cv::Mat drawImg;
	double P1[P1NUM];
	double P2[P2NUM];
	double*hist=(double *)malloc((NEIBOR+2)*BLOCKNUM*BLOCKNUM*sizeof(double));
	double*gabor=(double *)malloc((FANGXIANG*(CHIDU-CHIDU1)*2)*BLOCKNUM_GABOR*BLOCKNUM_GABOR*sizeof(double)); 
	if((cal_flag&0x01)==1)
	{
		//free(hist);
		
		lbpImg=lbp_hist(grayImg,hist,BLOCKNUM,BLOCKNUM,NEIBOR);
		//imshow("a", lbpImg);
		//cv::waitKey(0);
	}
	if((cal_flag&0x04)==4)
	{
		
		drawImg=shapeFeatureCal(bwImg,P1);
		calGLCM(grayImg,0,P2);
		/*for(int i=0;i<4;i++){
			cout<<P2[i]<<" ";
		}*/
	}
	
	if((cal_flag&0x02)==2)
	{
		//free(gabor);
		
		gaborImg=gabor_cal_all(gabor,grayImg);
	}
	int pos1=line.find('(');
	int pos2=line.rfind('\\',pos1);			
	string classname=line.substr(pos2+1,pos1-pos2);
	int classlabel=atoi(classname.c_str());
		cout <<"->finish!"<< endl; 
	if(oldclass!=classlabel)
	{
		oldclass=classlabel;
		writeid++;
		stringstream ss;
		ss<<classlabel;
		string str=".\\photosave\\gray"+ss.str();
		str+=".jpg";
		imwrite(str,grayImg);
		str=".\\photosave\\bw"+ss.str();
		str+=".jpg";
		imwrite(str,bwImg);
		if((cal_flag&0x04)==4){
			str=".\\photosave\\shape"+ss.str();
			str+=".jpg";
			imwrite(str,drawImg);
		}
		
		
		if((cal_flag&0x01)==1)
		{
			//imshow("a", lbpImg);
			//cv::waitKey(0);
			lbpImg.convertTo(lbpImg,CV_8UC1,1.0/255.0);
			//cv::normalize(lbpImg,lbpImg,0,255,CV_MINMAX,CV_8U);
			//imshow("a",lbpImg);
			//cv::waitKey(0);
			//ss<<classlabel;
			str=".\\photosave\\lbp"+ss.str();
			str+=".jpg";
			imwrite(str,lbpImg);
			
			/*x1<<writeid;
			for(int i=0;i<(NEIBOR+2)*BLOCKNUM*BLOCKNUM;i++)
			{
				x1<<","<<*(hist+i);
			}
			x1<<"\n";*/
		}
		if((cal_flag&0x02)==2)
		{			
			//ss<<classlabel;
			str=".\\photosave\\gabor"+ss.str();
			str+=".jpg";
			imwrite(str,gaborImg);
		}
		if(writeid!=1){
		cout<<"The "<<writeid-1<<"th kind of leaves have been calculated"<<endl;
		}
	}
	if((cal_flag&0x01)==1)
	{
		for(int i=0;i<(NEIBOR+2)*BLOCKNUM*BLOCKNUM;i++)
		{
			f<<*(hist+i)<<" ";
		}
			free(hist);
			f<<writeid<<'\n';
	}
	if((cal_flag&0x04)==4)
	{
		for(int i=0;i<P1NUM;i++)
		{
			f2<<P1[i]<<" ";
		}
		for(int i=0;i<P2NUM;i++)
		{
			f2<<P2[i]<<" ";
		}
			
		f2<<writeid<<'\n';
	}
	if((cal_flag&0x02)==2)
	{
		for(int i=0;i<(FANGXIANG*(CHIDU-CHIDU1)*2)*BLOCKNUM_GABOR*BLOCKNUM_GABOR;i++)
		{
			f1<<*(gabor+i)<<" ";
			
		}
		free(gabor);
		f1<<writeid<<'\n';


	}	
	
}