/*************************************************************************
    > File Name: test.cpp
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Tue 21 Mar 2017 02:36:44 PM CST
 ************************************************************************/

#include <iostream>
#include "opencv2/opencv.hpp"
#include "TextLineOCR.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[] ){
	Mat im;
	TextLineReader<float> r;
	//for(int i=0;i<1000;i++)
	cout<<r.read(im)<<endl;
	return 0;
}
