#include "lbp.h"
#include "include.h"

using namespace std;

int lmin(int a,int b);
int lmax(int a,int b);
 void getmapping(int neibor,int *table);
  cv::Mat elbp(cv::InputArray src, int radius, int neighbors) ;



// ����LBPM�Ŀռ�ֱ��ͼ�ֲ����õ�һ��һά����
// srcΪLBPM��ͨ��olbp����elbp����õ���
// numPatternsΪ����LBP��ģʽ��Ŀ��һ��Ϊ2����
// grid_x��grid_y�ֱ�Ϊÿ�л�ÿ�е�block����
// normedΪ�Ƿ���й�һ������
void hist_cell(cv::Mat grayImg,double*hist,int nei)
{
	int *table = (int *)malloc((int)pow(2.0,nei)*sizeof(int));
	getmapping(nei,table);
	/*ofstream ff("cell.txt");
	ff<<grayImg;
	ff.close();*/
	for(int i=0;i<nei+2;i++)
	{
		*(hist+i)=0;
	}
	
	for(int i=0;i<grayImg.rows;i++)
	{
		for(int j=0;j<grayImg.cols;j++)
		{
			grayImg.at<int>(i,j)=*(table+grayImg.at<int>(i,j));
			*(hist+grayImg.at<int>(i,j))+=1;
		}
	}
	for(int i=0;i<nei+2;i++)
	{
		*(hist+i)=*(hist+i)/(grayImg.rows*grayImg.cols);
	}
	free(table);
}
cv::Mat lbp_hist(cv::Mat grayImg,double*hist,int grid_x, int grid_y,int nei)
{
	grayImg=elbp(grayImg,1,nei);
	cv::Mat reImg;
	grayImg.copyTo(reImg);
	//grayImg.convertTo(grayImg,CV_8UC1,1.0/255.0);
	//imshow("source", grayImg);
	int width = grayImg.cols/grid_x;
    int height = grayImg.rows/grid_y;
//ofstream ff("hist.txt");
	 for(int i = 0; i < grid_y; i++) 
    {
        for(int j = 0; j < grid_x; j++)
         {
			int weight=0;
			weight=lmin(lmin(i-0,grid_y-i-1),lmin(j-0,grid_x-j-1))+1;
			//cout<<i<<' '<<j<<' '<<weight<<'\n';
			 // ��ȡָ������
            cv::Mat src_cell = cv::Mat(grayImg, cv::Range(i*height,(i+1)*height), cv::Range(j*width,(j+1)*width));
            // ����ָ�������ֱ��ͼ

			double*hist_c=(double *)malloc((nei+2)*sizeof(double));
            hist_cell(src_cell,hist_c,nei);
			//cout<<","<<(nei+2)*i*grid_x+j*(nei+2)+nei+2;
			
			for(int k=(nei+2)*i*grid_x+j*(nei+2);k<(nei+2)*i*grid_x+j*(nei+2)+nei+2;k++)
			{
				*(hist+k)=*(hist_c+k-(nei+2)*i*grid_x-j*(nei+2))*weight;
				//ff<<" "<<(float)*(hist+k);
			}
			//ff<<"\r\n";
			free(hist_c);
			
        }
    }
	// ff.close();
	return reImg;
	
}
static void dec2bin(int m,char *s,int n)
{
	for(int i=0;i<n;i++)
	{
		if((m&(1<<i))==(1<<i))
		{
			*(s+n-i-1)='1';
		}
		else
		{
			*(s+n-i-1)='0';
		}
	}
}
void getmapping(int neibor,int *table)
{
	//int *table = (int *)malloc((int)pow(2.0,neibor)*sizeof(int));
	//ofstream ff("table.txt");
	int newMax  = 0; //number of patterns in the resulting LBP code
	int index   = 0;
	newMax = neibor + 2;
	char*i_bin=(char *)malloc(neibor*sizeof(char));
	char*j_bin=(char *)malloc(neibor*sizeof(char));
	for(int i=0;i<(int)pow(2.0,neibor);i++)
	{
		
		dec2bin(i,i_bin,neibor);//ת��Ϊ2�����ַ���
		for(int j=0;j<neibor-1;j++)
		{
			*(j_bin+j)=*(i_bin+j+1);
		}
		*(j_bin+neibor-1)=*i_bin;//����
		int numt=0;
		for(int j=0;j<neibor;j++)
		{
			if(*(j_bin+j)!=*(i_bin+j))
			{
				numt++;
			}
		}
		if(numt<=2)
		{
			int sum=0;
			for(int j=0;j<neibor;j++)
			{
				sum+=*(i_bin+j)-'0';
			}
			*(table+i)=sum;
			
		}
		else
		{
			*(table+i)=neibor+1;
		}
		//ff<<*(table+i)<<" ";
	}
//	ff.close();

}
static cv::Mat spatial_histogram(cv::InputArray _src, int numPatterns,
                             int grid_x, int grid_y, bool normed)
{
    cv::Mat src = _src.getMat();
    // allocate memory for the spatial histogramΪLBPH�����ڴ�ռ�
    cv::Mat result = cv::Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
    // return matrix with zeros if no data was given�����û���������ݣ����ص���0
    if(src.empty())
        return result.reshape(1,1);
    // calculate LBP patch size block�ĳߴ�
    int width = src.cols/grid_x;
    int height = src.rows/grid_y;
    // initial result_row ��ʼ�������
    int resultRowIdx = 0;
    // iterate through grid
    for(int i = 0; i < grid_y; i++) 
    {
        for(int j = 0; j < grid_x; j++)
         {
            // ��ȡָ������
            cv::Mat src_cell = cv::Mat(src, cv::Range(i*height,(i+1)*height), cv::Range(j*width,(j+1)*width));
            // ����ָ�������ֱ��ͼ
			
            //cv::Mat cell_hist = histc(src_cell, 0, (numPatterns-1), true);
            // copy to the result matrix ������õ��Ľ��������ÿһ��
            cv::Mat result_row = result.row(resultRowIdx);
            //cell_hist.reshape(1,1).convertTo(result_row, CV_32FC1);
            // increase row count in result matrix
            resultRowIdx++;
        }
    }
    // return result as reshaped feature vector
    return result.reshape(1,1);
}

template <typename _Tp> static
inline void elbp_(cv::InputArray _src, cv::OutputArray _dst, int radius, int neighbors)
{
    //get matrices
    cv::Mat src = _src.getMat();
    // allocate memory for result��˲������ⲿ��_dst�����ڴ�ռ䣬����������Ͷ���int
    _dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    cv::Mat dst = _dst.getMat();
    // zero
    dst.setTo(0);
    for(int n=0; n<neighbors; n++) 
    {
        // sample points ��ȡ��ǰ������
        float x = static_cast<float>(-radius) * sin(2.0*CV_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * cos(2.0*CV_PI*n/static_cast<float>(neighbors));
        // relative indices ��ȡ������ȡ��
        int fx = static_cast<int>(floor(x)); // ����ȡ��
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));  // ����ȡ��
        int cy = static_cast<int>(ceil(y));
        // fractional part С������
        float tx = x - fx;
        float ty = y - fy;
        // set interpolation weights �����ĸ���Ĳ�ֵȨ��
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data ѭ������ͼ������
        for(int i=radius; i < src.rows-radius;i++) 
        {
            for(int j=radius;j < src.cols-radius;j++) 
            {
                // calculate interpolated value �����ֵ��t��ʾ�ĸ����Ȩ�غ�
                float t = w1*src.at<_Tp>(i+fy,j+fx) + 
w2*src.at<_Tp>(i+fy,j+cx) + 
w3*src.at<_Tp>(i+cy,j+fx) + 
w4*src.at<_Tp>(i+cy,j+cx);
                // floating point precision, so check some machine-dependent epsilon
                // std::numeric_limits<float>::epsilon()=1.192092896e-07F
                // ��t>=src(i,j)��ʱ��ȡ1����������Ӧ����λ
                dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || 
                            (std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

static void elbp(cv::InputArray src, cv::OutputArray dst, int radius, int neighbors)
{
    int type = src.type();
    switch (type) {
    case CV_8SC1:   elbp_<char>(src,dst, radius, neighbors); break;
    case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
    case CV_16SC1:  elbp_<short>(src,dst, radius, neighbors); break;
    case CV_16UC1:  elbp_<unsigned short>(src,dst, radius, neighbors); break;
    case CV_32SC1:  elbp_<int>(src,dst, radius, neighbors); break;
    case CV_32FC1:  elbp_<float>(src,dst, radius, neighbors); break;
    case CV_64FC1:  elbp_<double>(src,dst, radius, neighbors); break;
    default:
        string error_msg = cv::format("Using Circle Local Binary Patterns for feature extraction only works                                     on single-channel images (given %d). Please pass the image data as a grayscale image!", type);
        CV_Error(CV_StsNotImplemented, error_msg);
        break;
    }
}
cv::Mat elbp(cv::InputArray src, int radius, int neighbors) {
    cv::Mat dst;
    elbp(src, dst, radius, neighbors);
    return dst;
}
int lmin(int a,int b)
{
	if(a>b)
		return b;
	else
		return a;
}
int lmax(int a,int b)
{
	if(a<b)
		return b;
	else
		return a;
}