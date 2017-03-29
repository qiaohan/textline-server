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
	ifstream dict_file( (binpath+"all_class_random.txt").c_str() );
	string word;
	while(getline(dict_file, word))
	{
		cout<<word<<endl;
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
	vector<int> t = crnn -> predict(NULL);
	string out;
	cJSON *root,*status,*result;
	root=cJSON_CreateObject();
	cJSON_AddNumberToObject(root,"status",1);
	result=cJSON_CreateArray();
	std::stringstream buf;
	buf.str(std::string());
	buf.clear();
	buf<<"meiwenzi:";
	//buf<<feature[2];
	buf<<";";
	buf<<"youwenzi:";
	//buf<<feature[1]+feature[0];
	buf<<";";
	buf<<"guanggao:";
	//buf<<feature[0];
	buf<<";";
	buf<<"feiguanggao:";
	//buf<<feature[1];
	string result_str=buf.str();
	cJSON_AddItemToObject(root,"result",cJSON_CreateString(result_str.c_str()));
	out=cJSON_Print(root);
	cJSON_Delete(root);
	return out;
}

#endif
