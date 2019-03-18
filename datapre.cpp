#include "datapre.h" 
#include "tinyxml.h"
#include "pca.h"
//#include <contrib\contrib.hpp>
#include<iomanip>
#include "datasvm.h"

//#include<windows.h>


int DataLoadAll(int lbpBlockNum,int gaborBlockNum,int gaborF,int gaborDur,int cal_flag)
{
	cv::Mat studyMat,lbpMat,gaborMat;
	cv::Mat label;
	int studyRow=countFileLine("study.txt");
	int lbpCol=lbpBlockNum*lbpBlockNum*(NEIBOR+2);
	int gaborCol=gaborBlockNum*gaborBlockNum*gaborF*gaborDur*2;
	if((cal_flag&1)==1)	
		LoadData("lbp.txt",lbpMat,studyRow,lbpCol+1,1);
	if((cal_flag&2)==2)
		LoadData("gabor.txt",gaborMat,studyRow,gaborCol+1,1);
	if((cal_flag&4)==4)
		LoadData("study.txt",studyMat,studyRow,P1NUM+P2NUM+1,1);

	if ((cal_flag&4)==4)
	{
		label=studyMat(cv::Rect(P1NUM+P2NUM,0,1,studyRow));
		studyMat=studyMat(cv::Rect(0,0,P1NUM+P2NUM,studyRow)).t();
	}
	if((cal_flag&1)==1)
	{
		lbpMat=lbpMat(cv::Rect(0,0,lbpCol,studyRow)).t();
		if((cal_flag&4)==0)
		{
			label=lbpMat(cv::Rect(lbpCol,0,1,studyRow));
		}
	}
	if((cal_flag&2)==2)
	{
		gaborMat=gaborMat(cv::Rect(0,0,gaborCol,studyRow)).t();
		if((cal_flag&4)==0&&(cal_flag&1)==0)
		{
			label=gaborMat(cv::Rect(gaborCol,0,1,studyRow));
		}
	}
	cv::Mat AllData;
	if((cal_flag&4)==4)	
		AllData.push_back(studyMat);
	if((cal_flag&1)==1)
		AllData.push_back(lbpMat);
	if((cal_flag&2)==2)
		AllData.push_back(gaborMat);

	
	
	AllData=AllData.t();
	WriteData("all_data.txt", AllData);
	cout<<"AllData-->cols:"<<AllData.cols<<",rows:"<<AllData.rows<<endl;
	cout<<"Scale from 0 to 1...";
	ScaleNorm(AllData);
	WriteData("norm_data.txt", AllData);
	WriteData("labels.txt",label);
	cout<<"finish"<<endl;
	
	exeMatlabDATA2SVM();
	//WinExec("data2svm.exe",SW_SHOW);
	//exec("data2svm.exe");
	cout<<"pca & lda finish"<<endl;
	//pcaMain(AllData);
/*
	// perform PCA
    PCA pca(AllData, cv::Mat(), CV_PCA_DATA_AS_ROW, 0.999);
	cout<<"?";
	WriteData("savedata\\pca_mean.txt", pca.mean);
	WriteData("savedata\\pca_eigenvectors.txt", pca.eigenvectors);
	cv::Mat pcaAlldata=pca.project(AllData);
	//WriteData("pca_data.txt", pcaAlldata);
	cout<<"After PCA, AllData-->cols:"<<pcaAlldata.cols<<",rows:"<<pcaAlldata.rows<<endl;
	
	//LDA
	LDA lda(pcaAlldata,label);
	WriteData("savedata\\lda_eigenvectors.txt", lda.eigenvectors());
	cv::Mat ldaAlldata=lda.project(pcaAlldata);
	cout<<"After LDA, AllData-->cols:"<<ldaAlldata.cols<<",rows:"<<ldaAlldata.rows<<endl;
	*/
	return 0;
}
void ScaleNorm(cv::Mat& src)
{
	src=src.t();
	int rows=src.rows;
	int cols=src.cols;
	
	vector<float> max;
	vector<float> min;
	//cout<<"00.00%";
	for(int i=0;i<rows;i++)
	{
		float temp_max=std::numeric_limits<float>::min();
		float temp_min=std::numeric_limits<float>::max();
		float *data=src.ptr<float>(i);
		//cout<<"\b\b\b\b\b\b"<<setprecision(2)<<(float)i/rows/2*100<<"%";
		for(int j=0;j<cols;j++)
		{
			if(data[j]>temp_max)
			{
				temp_max=data[j];
			}
			if(data[j]<temp_min)
			{
				temp_min=data[j];
			}
		}
		max.push_back(temp_max);
		min.push_back(temp_min);
	}
	cv::Mat rangeMat;
	cv::Mat minMat;
	
	for(int i=0;i<rows;i++)
	{
		//cout<<"\b\b\b\b\b\b"<<setprecision(2)<<50.0+(float)i/rows/2*100<<"%";
		cv::Mat tempRangeMat(1,1,CV_32FC1),tempMinMat(1,1,CV_32FC1);
		tempMinMat.at<float>(0,0)=min[i];
		repeat(tempMinMat,1,cols,tempMinMat);
		if(max[i]==min[i])
		{
			tempRangeMat.at<float>(0,0)=1.0;
		}
		else
		{
			tempRangeMat.at<float>(0,0)=1.0/(max[i]-min[i]);
		}
		repeat(tempRangeMat,1,cols,tempRangeMat);
		rangeMat.push_back(tempRangeMat);
		minMat.push_back(tempMinMat);
	}
	ofstream minf("savedata\\scale_min");
	ofstream maxf("savedata\\scale_max");
	for(int i=0;i<min.size();i++)
	{
		minf<<min[i]<<" ";
		maxf<<max[i]<<" ";
	}
	minf.close();
	maxf.close();
	/*WriteData("minMat.txt", minMat);
	WriteData("rangeMat.txt", rangeMat);*/
	src=src-minMat;
	src=src.mul(rangeMat);
	src=src.t();
}
settingdata ReadSetting()
{
	settingdata readdata;
	fstream f;
	f.open("Setting.xml",ios::in);
	if(!f)
	{
		readdata.cal_flag=0;
		readdata.gaborBlockNum=0;
		readdata.gaborDur=0;
		readdata.gaborF=0;
		readdata.lbpBlockNum=0;
		return readdata;
	}
	f.close();
	//创建一个XML的文档对象。
    TiXmlDocument *myDocument = new TiXmlDocument("Setting.xml");
    myDocument->LoadFile();
    //获得根元素，即Persons。
    TiXmlElement *RootElement = myDocument->RootElement();
    //输出根元素名称，即输出Persons。
    
    //获得第一个节点。
    TiXmlElement *lbpBlockNum = RootElement->FirstChildElement();
	TiXmlElement *gaborBlockNum = lbpBlockNum->NextSiblingElement();
	TiXmlElement *gaborF = gaborBlockNum->NextSiblingElement();
	TiXmlElement *gaborD = gaborF->NextSiblingElement();
	TiXmlElement *select = gaborD->NextSiblingElement();

    //输出第一个Person的name内容，即周星星；age内容，即；ID属性，即。
    string str1= lbpBlockNum->FirstChild()->Value();
	readdata.lbpBlockNum=atoi(str1.c_str());
	str1= gaborBlockNum->FirstChild()->Value();
	readdata.gaborBlockNum=atoi(str1.c_str());
	str1= gaborF->FirstChild()->Value();
	readdata.gaborF=atoi(str1.c_str());
	str1= gaborD->FirstChild()->Value();
	readdata.gaborDur=atoi(str1.c_str());
	str1= select->FirstChild()->Value();
	readdata.cal_flag=atoi(str1.c_str());
	return readdata;
}

int SaveSetting(int lbpBlockNum,int gaborBlockNum,int gaborF,int gaborDur,int cal_flag)
{
	//创建一个XML的文档对象
	TiXmlDocument *myDocument = new TiXmlDocument();
	
    //创建一个根元素并连接。
    TiXmlElement *RootElement = new TiXmlElement("Settings");
    myDocument->LinkEndChild(RootElement);
    //创建一个Person元素并连接。
    TiXmlElement *PersonElement = new TiXmlElement("lbp_block_num");
    RootElement->LinkEndChild(PersonElement);
	TiXmlElement *PersonElement1 = new TiXmlElement("gabor_block_num");
    RootElement->LinkEndChild(PersonElement1);
	TiXmlElement *PersonElement2 = new TiXmlElement("gabor_f");
    RootElement->LinkEndChild(PersonElement2);
	TiXmlElement *PersonElement3 = new TiXmlElement("gabor_d");
    RootElement->LinkEndChild(PersonElement3);
	TiXmlElement *PersonElement4 = new TiXmlElement("select");
    RootElement->LinkEndChild(PersonElement4);
    ////设置Person元素的属性。
    //PersonElement->SetAttribute("ID", "1");
    //创建name元素、age元素并连接。
   /* TiXmlElement *NameElement = new TiXmlElement("name");
    TiXmlElement *AgeElement = new TiXmlElement("age");
    PersonElement->LinkEndChild(NameElement);
    PersonElement->LinkEndChild(AgeElement);*/
    //设置name元素和age元素的内容并连接。
	stringstream ss;
	ss<<lbpBlockNum;
	TiXmlText *lbp_block_num = new TiXmlText(ss.str().c_str());
	
	ss.str("");
	ss<<gaborBlockNum;
	TiXmlText *gabor_block_num = new TiXmlText(ss.str().c_str());

	ss.str("");
	ss<<gaborF;
	TiXmlText *gabor_f = new TiXmlText(ss.str().c_str());

	ss.str("");
	ss<<gaborDur;
	TiXmlText *gabor_d = new TiXmlText(ss.str().c_str());

	ss.str("");
	ss<<cal_flag;
	TiXmlText *select = new TiXmlText(ss.str().c_str());

    PersonElement->LinkEndChild(lbp_block_num);
	PersonElement1->LinkEndChild(gabor_block_num);
	PersonElement2->LinkEndChild(gabor_f);
	PersonElement3->LinkEndChild(gabor_d);
	PersonElement4->LinkEndChild(select);
    myDocument->SaveFile("Setting.xml");//保存到文件
	return 0;
}
/*---------------------------- 
 * 功能 : 从 .txt 文件中读入数据，保存到 cv::cv::Mat 矩阵 
 *      - 默认按 double 格式读入数据， 
 *      - 如果没有指定矩阵的行、列和通道数，则输出的矩阵是单通道、N 行 1 列的 
 *---------------------------- 
 * 函数 : LoadData 
 * 访问 : public  
 * 返回 : -1：打开文件失败；0：按设定的矩阵参数读取数据成功；1：按默认的矩阵参数读取数据 
 * 
 * 参数 : fileName    [in]    文件名 
 * 参数 : matData [out]   矩阵数据 
 * 参数 : matRows [in]    矩阵行数，默认为 0 
 * 参数 : matCols [in]    矩阵列数，默认为 0 
 * 参数 : matChns [in]    矩阵通道数，默认为 0 
 */  
int LoadData(string fileName, cv::Mat& matData, int matRows , int matCols, int matChns )  
{  
    int retVal = 0;  
  
    // 打开文件  
    ifstream inFile(fileName.c_str(), ios_base::in);  
    if(!inFile.is_open())  
    {  
        cout << "读取文件失败" << endl;  
        retVal = -1;  
        return (retVal);  
    }  
  
    // 载入数据  
    istream_iterator<float> begin(inFile);    //按 float 格式取文件数据流的起始指针  
    istream_iterator<float> end;          //取文件流的终止位置  
    vector<float> inData(begin,end);      //将文件数据保存至 std::vector 中  
    cv::Mat tmpMat = cv::Mat(inData);       //将数据由 std::vector 转换为 cv::Mat  
  
    // 输出到命令行窗口  
    //copy(vec.begin(),vec.end(),ostream_iterator<double>(cout,"\t"));   
  
    // 检查设定的矩阵尺寸和通道数  
    size_t dataLength = inData.size();  
    //1.通道数  
    if (matChns == 0)  
    {  
        matChns = 1;  
    }  
    //2.行列数  
    if (matRows != 0 && matCols == 0)  
    {  
        matCols = dataLength / matChns / matRows;  
    }   
    else if (matCols != 0 && matRows == 0)  
    {  
        matRows = dataLength / matChns / matCols;  
    }  
    else if (matCols == 0 && matRows == 0)  
    {  
        matRows = dataLength / matChns;  
        matCols = 1;  
    }  
    //3.数据总长度  
    if (dataLength != (matRows * matCols * matChns))  
    {  
        cout << "读入的数据长度 不满足 设定的矩阵尺寸与通道数要求，将按默认方式输出矩阵！" << endl;  
        retVal = 1;  
        matChns = 1;  
        matRows = dataLength;  
    }   
  
    // 将文件数据保存至输出矩阵  
    matData = tmpMat.reshape(matChns, matRows).clone();  
      
    return (retVal);  
} 
int countFileLine(string str)
{

	ifstream file(str.c_str());
    
    int count = 0;
    while (file) {
        getline(file, str);//从文件中读取一行
        remove(str.begin(), str.end(), ' ');//这个算法函数在algorithm头文件中，删除一行中的空格
        remove(str.begin(), str.end(), '\t');//删除一行中的制表符，因为制表符和空格都是空的
        if (str.length() > 0) {//如果删除制表符和空格之后的一行数据还有其他字符就算有效行
            count ++;
        }
    }
    
   //cout<<count;
    return count;

}
/*---------------------------- 
 * 功能 : 将 cv::Mat 数据写入到 .txt 文件 
 *---------------------------- 
 * 函数 : WriteData 
 * 访问 : public  
 * 返回 : -1：打开文件失败；0：写入数据成功；1：矩阵为空 
 * 
 * 参数 : fileName    [in]    文件名 
 * 参数 : matData [in]    矩阵数据 
 */  
int WriteData(string fileName, cv::Mat& matData)  
{  
    int retVal = 0;  
  
    // 检查矩阵是否为空  
    if (matData.empty())  
    {  
        cout << "矩阵为空" << endl;   
        retVal = 1;  
        return (retVal);  
    }  
  
    // 打开文件  
    ofstream outFile(fileName.c_str(), ios_base::out);  //按新建或覆盖方式写入  
    if (!outFile.is_open())  
    {  
        cout << "打开文件失败" << endl;   
        retVal = -1;  
        return (retVal);  
    }  
  
    // 写入数据  
    for (int r = 0; r < matData.rows; r++)  
    {  
        for (int c = 0; c < matData.cols; c++)  
        {  
            float data = matData.at<float>(r,c);    //读取数据，at<type> - type 是矩阵元素的具体数据格式  
            outFile << data << "\t" ;   //每列数据用 tab 隔开  
        }  
        outFile << endl;  //换行  
    }  
  
    return (retVal);  
}


