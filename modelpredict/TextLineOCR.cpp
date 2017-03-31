/*************************************************************************
    > File Name: TextLineOCR.cpp
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Tue 21 Mar 2017 10:40:13 AM CST
 ************************************************************************/
#ifndef __TEXTLINEOCR_CPP__
#define __TEXTLINEOCR_CPP__

#include "cJSON.h"
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;


template<typename T>
TextLineReader<T>::TextLineReader(string binpath){
	crnn = new CRNN<T>(binpath,0);
	ifstream dict_file( (binpath+"word_dict.txt").c_str() );
	string word;
	while(getline(dict_file, word))
	{
		cout<<word_dict.size()<<':'<<word<<endl;
		word_dict.push_back(word);
	}
}

template<typename T>
TextLineReader<T>::~TextLineReader(){
	delete crnn;
}
/*
template<typename T>
vector<int> TextLineReader<T>::read(Mat im){
	vector<int> t = crnn -> predict(NULL);
	return t;
}
*/
template<typename T>
string TextLineReader<T>::read(Mat im){
	if(im.type()!=CV_8UC3 || im.cols<im.rows)
	{
		cout<<"image type:"<<im.type()<<endl;
		return "";
	}
	//im = im.t();
	Mat im_resized,tmp;
	Mat im_float = Mat::zeros(IMAGE_H,IMAGE_W,CV_32FC3);
	int newh = IMAGE_H;
	int neww = im.cols/im.rows * newh;
	//cout<<"old w:"<<im.cols<<"old h:"<<im.rows<<endl;
	//cout<<"new w:"<<neww<<"new h:"<<newh<<endl;
	resize(im,im_resized,Size(neww,newh));
	im_resized.convertTo(tmp,CV_32FC3);
	if(neww>IMAGE_W)
	{
		neww = IMAGE_W;
		tmp(Rect(0,0,IMAGE_W,IMAGE_H)).copyTo( im_float );
	}
	else
		tmp.copyTo( im_float(Rect(0,0,neww,IMAGE_H)) );

	char * imgd = (char*)malloc(im_float.cols*im_float.rows*im_float.channels()*sizeof(float));
	vector<Mat> input_channels;
	split(im_float,input_channels);
	memcpy(imgd+0*im_float.cols*im_float.rows*sizeof(float),input_channels[0].data,im_float.cols*im_float.rows*sizeof(float));
	memcpy(imgd+1*im_float.cols*im_float.rows*sizeof(float),input_channels[1].data,im_float.cols*im_float.rows*sizeof(float));
	memcpy(imgd+2*im_float.cols*im_float.rows*sizeof(float),input_channels[2].data,im_float.cols*im_float.rows*sizeof(float));

	//cout<<input_channels[0].row(0)<<endl;
	//for(int i=0; i<100; i++)
	//	cout<<*( (float*)(imgd+i) );
	cout<<endl;
	vector<int> t = crnn -> predict(imgd,neww);
	free(imgd);
	
	string out;
	cJSON *root,*status,*result;
	root=cJSON_CreateObject();
	cJSON_AddNumberToObject(root,"status",1);
	result=cJSON_CreateArray();
	std::stringstream buf;
	buf.str(std::string());
	buf.clear();
	
	for(int i=0; i<t.size(); i++){
		//cout<<t[i]<<',';
		buf<<word_dict[t[i]];
	}

	string result_str=buf.str();
	cJSON_AddItemToObject(root,"text",cJSON_CreateString(result_str.c_str()));
	out=cJSON_Print(root);
	cJSON_Delete(root);
	return out;
}

#endif
