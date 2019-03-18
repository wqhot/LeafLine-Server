#include "datasvm.h"
#include <opencv2/ml/ml.hpp>
#include "datapre.h"
int readdata(string FileName)
{
	//string fileName="data2svm.txt";	
	int matlength=countFileLine(FileName);
	int matwidth=countFileCol(FileName);
	//cout<<"length:"<<matlength<<" width:"<<matwidth<<endl;
	cv::Mat svmMatAll,svmMat,label;
	LoadData(FileName,svmMatAll,matlength,matwidth,1);
	label=svmMatAll(cv::Rect(matwidth-1,0,1,matlength));
	svmMat=svmMatAll(cv::Rect(0,0,matwidth-1,matlength));
	//svm·ÖÀà
	//ml::SVM::Params params;
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::Types::C_SVC);
	svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
	svm->setGamma(0.0125);
	svm->setC(32);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
	//params.svm_type= CvSVM::C_SVC;
	//params.kernel_type= CvSVM::RBF;
	svm->train(svmMat, cv::ml::SampleTypes::ROW_SAMPLE, label);
	//svm->train(svmMat,label,Mat(),Mat(),params);
	svm->save("svm_train_opencv.xml");
	//SVM.save("svm_train_opencv.xml");
	
	cv::Mat labels;
	svm->predict(svmMat, label);
	//SVM.predict(svmMat,labels);

	//cout<<labels.t();
    return 0;
} 

int countFileCol(string fileName)
{
	ifstream f(fileName.c_str());
	char a[1000];
	f.getline(a,1000,'\n');
	int i=0,count=0;
	while(*(a+i))
	{
		if(*(a+i)==' ')
		{
			count++;
		}
		i++;
	}
	return count+1;
}
