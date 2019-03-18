#ifndef  __DATAPRE_H_
#define  __DATAPRE_H_
#include "include.h"
#include <iterator> 
struct settingdata{
	int lbpBlockNum;
	int gaborBlockNum;
	int gaborF;
	int gaborDur;
	int cal_flag;
};
void ScaleNorm(cv::Mat& src);
int LoadData(string fileName, cv::Mat& matData, int matRows = 0, int matCols = 0, int matChns = 0);
int countFileLine(string str);
int WriteData(string fileName, cv::Mat& matData) ;
int DataLoadAll(int lbpBlockNum,int gaborBlockNum,int gaborF,int gaborDur,int cal_flag);
int SaveSetting(int lbpBlockNum,int gaborBlockNum,int gaborF,int gaborDur,int cal_flag);
settingdata ReadSetting();
#endif