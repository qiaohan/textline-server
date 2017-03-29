#ifndef _REQP_H_
#define _REQP_H_
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct ReqPackage
{
//ReqPackage(cv::Mat m){src_mat = m;}
	string imgPath;
	cv::Mat src_mat;
	//string status;
	//string res;
	//int flag;
	void* id;
	//void* callback(void* ptr)=NULL;
};
struct ResPackage
{
	vector<int> chars;
};

#endif
