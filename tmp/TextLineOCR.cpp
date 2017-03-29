/*************************************************************************
    > File Name: TextLineOCR.cpp
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Tue 21 Mar 2017 10:40:13 AM CST
 ************************************************************************/

#include<iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;


template<typename T>
TextLineReader<T>::TextLineReader(string binpath){

	const char *conv1_bin = "conv1_weights.bin";
	const char *conv1_bias_bin = "conv1_bias.bin";
	const char *conv2_bin = "conv2_weights.bin";
	const char *conv2_bias_bin = "conv2_bias.bin";
	const char *conv3_bin = "conv3_weights.bin";
	const char *conv3_bias_bin = "conv3_bias.bin";
	const char *conv4_bin = "conv4_weights.bin";
	const char *conv4_bias_bin = "conv4_bias.bin";
	const char *conv5_bin = "conv5_weights.bin";
	const char *conv5_bias_bin = "conv5_bias.bin";
	const char *conv6_bin = "conv6_weights.bin";
	const char *conv6_bias_bin = "conv6_bias.bin";
	const char *conv7_bin = "conv7_weights.bin";
	const char *conv7_bias_bin = "conv7_bias.bin";
	const char *ip_bin = "logit_weights.bin";
	const char *ip_bias_bin = "logit_bias.bin";

	char argv[] = "crnn";
    Layer_t<T>* conv1 = new Layer_t<T>(3,64,3,conv1_bin,conv1_bias_bin,argv);
		layers_param.push_back(conv1);
		std::cout<<"conv1 finished"<<std::endl;
	Layer_t<T>* conv2 = new Layer_t<T>(64,128,3,conv2_bin,conv2_bias_bin,argv);
		layers_param.push_back(conv2);
	Layer_t<T>* conv3 = new Layer_t<T>(128,256,3,conv3_bin,conv3_bias_bin,argv);
		layers_param.push_back(conv3);
    Layer_t<T>* conv4 = new Layer_t<T>(256,256,3,conv4_bin,conv4_bias_bin,argv);
		layers_param.push_back(conv4);
    Layer_t<T>* conv5 = new Layer_t<T>(256,512,3,conv5_bin,conv5_bias_bin,argv);
		layers_param.push_back(conv5);
    Layer_t<T>* conv6 = new Layer_t<T>(512,512,3,conv6_bin,conv6_bias_bin,argv);
		layers_param.push_back(conv6);
    Layer_t<T>* conv7 = new Layer_t<T>(512,512,3,conv7_bin,conv7_bias_bin,argv);
		layers_param.push_back(conv7);
    Layer_t<T>* ip = new Layer_t<T>(2048,3851,1,ip_bin,ip_bias_bin,argv);
		layers_param.push_back(ip);
	crnn = new network_t<T>(layers_param);
}

template<typename T>
TextLineReader<T>::~TextLineReader(){
	for(int i=0; i<layers_param.size(); i++)
		delete layers_param[i];
	delete crnn;
}

template<typename T>
vector<int> TextLineReader<T>::read(Mat im){
	return crnn -> classify_example(NULL);
}
