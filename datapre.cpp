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
	//����һ��XML���ĵ�����
    TiXmlDocument *myDocument = new TiXmlDocument("Setting.xml");
    myDocument->LoadFile();
    //��ø�Ԫ�أ���Persons��
    TiXmlElement *RootElement = myDocument->RootElement();
    //�����Ԫ�����ƣ������Persons��
    
    //��õ�һ���ڵ㡣
    TiXmlElement *lbpBlockNum = RootElement->FirstChildElement();
	TiXmlElement *gaborBlockNum = lbpBlockNum->NextSiblingElement();
	TiXmlElement *gaborF = gaborBlockNum->NextSiblingElement();
	TiXmlElement *gaborD = gaborF->NextSiblingElement();
	TiXmlElement *select = gaborD->NextSiblingElement();

    //�����һ��Person��name���ݣ��������ǣ�age���ݣ�����ID���ԣ�����
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
	//����һ��XML���ĵ�����
	TiXmlDocument *myDocument = new TiXmlDocument();
	
    //����һ����Ԫ�ز����ӡ�
    TiXmlElement *RootElement = new TiXmlElement("Settings");
    myDocument->LinkEndChild(RootElement);
    //����һ��PersonԪ�ز����ӡ�
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
    ////����PersonԪ�ص����ԡ�
    //PersonElement->SetAttribute("ID", "1");
    //����nameԪ�ء�ageԪ�ز����ӡ�
   /* TiXmlElement *NameElement = new TiXmlElement("name");
    TiXmlElement *AgeElement = new TiXmlElement("age");
    PersonElement->LinkEndChild(NameElement);
    PersonElement->LinkEndChild(AgeElement);*/
    //����nameԪ�غ�ageԪ�ص����ݲ����ӡ�
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
    myDocument->SaveFile("Setting.xml");//���浽�ļ�
	return 0;
}
/*---------------------------- 
 * ���� : �� .txt �ļ��ж������ݣ����浽 cv::cv::Mat ���� 
 *      - Ĭ�ϰ� double ��ʽ�������ݣ� 
 *      - ���û��ָ��������С��к�ͨ������������ľ����ǵ�ͨ����N �� 1 �е� 
 *---------------------------- 
 * ���� : LoadData 
 * ���� : public  
 * ���� : -1�����ļ�ʧ�ܣ�0�����趨�ľ��������ȡ���ݳɹ���1����Ĭ�ϵľ��������ȡ���� 
 * 
 * ���� : fileName    [in]    �ļ��� 
 * ���� : matData [out]   �������� 
 * ���� : matRows [in]    ����������Ĭ��Ϊ 0 
 * ���� : matCols [in]    ����������Ĭ��Ϊ 0 
 * ���� : matChns [in]    ����ͨ������Ĭ��Ϊ 0 
 */  
int LoadData(string fileName, cv::Mat& matData, int matRows , int matCols, int matChns )  
{  
    int retVal = 0;  
  
    // ���ļ�  
    ifstream inFile(fileName.c_str(), ios_base::in);  
    if(!inFile.is_open())  
    {  
        cout << "��ȡ�ļ�ʧ��" << endl;  
        retVal = -1;  
        return (retVal);  
    }  
  
    // ��������  
    istream_iterator<float> begin(inFile);    //�� float ��ʽȡ�ļ�����������ʼָ��  
    istream_iterator<float> end;          //ȡ�ļ�������ֹλ��  
    vector<float> inData(begin,end);      //���ļ����ݱ����� std::vector ��  
    cv::Mat tmpMat = cv::Mat(inData);       //�������� std::vector ת��Ϊ cv::Mat  
  
    // ����������д���  
    //copy(vec.begin(),vec.end(),ostream_iterator<double>(cout,"\t"));   
  
    // ����趨�ľ���ߴ��ͨ����  
    size_t dataLength = inData.size();  
    //1.ͨ����  
    if (matChns == 0)  
    {  
        matChns = 1;  
    }  
    //2.������  
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
    //3.�����ܳ���  
    if (dataLength != (matRows * matCols * matChns))  
    {  
        cout << "��������ݳ��� ������ �趨�ľ���ߴ���ͨ����Ҫ�󣬽���Ĭ�Ϸ�ʽ�������" << endl;  
        retVal = 1;  
        matChns = 1;  
        matRows = dataLength;  
    }   
  
    // ���ļ����ݱ������������  
    matData = tmpMat.reshape(matChns, matRows).clone();  
      
    return (retVal);  
} 
int countFileLine(string str)
{

	ifstream file(str.c_str());
    
    int count = 0;
    while (file) {
        getline(file, str);//���ļ��ж�ȡһ��
        remove(str.begin(), str.end(), ' ');//����㷨������algorithmͷ�ļ��У�ɾ��һ���еĿո�
        remove(str.begin(), str.end(), '\t');//ɾ��һ���е��Ʊ������Ϊ�Ʊ���Ϳո��ǿյ�
        if (str.length() > 0) {//���ɾ���Ʊ���Ϳո�֮���һ�����ݻ��������ַ�������Ч��
            count ++;
        }
    }
    
   //cout<<count;
    return count;

}
/*---------------------------- 
 * ���� : �� cv::Mat ����д�뵽 .txt �ļ� 
 *---------------------------- 
 * ���� : WriteData 
 * ���� : public  
 * ���� : -1�����ļ�ʧ�ܣ�0��д�����ݳɹ���1������Ϊ�� 
 * 
 * ���� : fileName    [in]    �ļ��� 
 * ���� : matData [in]    �������� 
 */  
int WriteData(string fileName, cv::Mat& matData)  
{  
    int retVal = 0;  
  
    // �������Ƿ�Ϊ��  
    if (matData.empty())  
    {  
        cout << "����Ϊ��" << endl;   
        retVal = 1;  
        return (retVal);  
    }  
  
    // ���ļ�  
    ofstream outFile(fileName.c_str(), ios_base::out);  //���½��򸲸Ƿ�ʽд��  
    if (!outFile.is_open())  
    {  
        cout << "���ļ�ʧ��" << endl;   
        retVal = -1;  
        return (retVal);  
    }  
  
    // д������  
    for (int r = 0; r < matData.rows; r++)  
    {  
        for (int c = 0; c < matData.cols; c++)  
        {  
            float data = matData.at<float>(r,c);    //��ȡ���ݣ�at<type> - type �Ǿ���Ԫ�صľ������ݸ�ʽ  
            outFile << data << "\t" ;   //ÿ�������� tab ����  
        }  
        outFile << endl;  //����  
    }  
  
    return (retVal);  
}


