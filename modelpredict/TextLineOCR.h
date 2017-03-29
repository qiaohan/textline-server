/*************************************************************************
    > File Name: TextLineOCR.h
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Tue 21 Mar 2017 10:35:56 AM CST
 ************************************************************************/
#ifndef __TEXTLINEOCR_H__
#define __TEXTLINEOCR_H__

#include <iostream>
#include "opencv2/opencv.hpp"
#include "crnn.h"
using namespace std;
using namespace cv;

template<typename T>
class TextLineReader{
	public:
		TextLineReader(string binpath="bin/");
		~TextLineReader();
		//vector<int> read(Mat im);
		string read(Mat im);
	private:
		vector<string> word_dict;
		CRNN<T> * crnn;
};

#include "TextLineOCR.cpp"
#endif
