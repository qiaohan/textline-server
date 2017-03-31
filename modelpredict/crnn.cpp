/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
 * This example demonstrates how to use CUDNN library to implement forward
 * pass. The sample loads weights and biases from trained network,
 * takes a few images of digits and recognizes them. The network was trained on 
 * the MNIST dataset using Caffe. The network consists of two 
 * convolution layers, two pooling layers, one relu and two 
 * fully connected layers. Final layer gets processed by Softmax. 
 * cublasSgemv is used to implement fully connected layers.

 * The sample can work in single, double, half precision, but it
 * assumes the data in files is stored in single precision
 */

#ifndef __CRNN_CPP__
#define __CRNN_CPP__

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <vector>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <cublas_v2.h>

//#include "ImageIO.h"
//#include "fp16_dev.h"
//#include "fp16_emu.h"
//#include "gemv.h"
#include "error_util.h"
#include "decode.h"

#define IMAGE_H 32
#define IMAGE_W 512
#define LOGITNUM 3851

template <typename value_type>
CRNN<value_type>::CRNN(std::string binpath, int devnum)
{
	checkCudaErrors( cudaSetDevice(devnum) );
	
	layers_param.clear();
    convs.clear();
	pools.clear();
	bottom.clear();
	top.clear();
	
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
	

	const char * argv = binpath.c_str();
    Layer_t<value_type>* conv1 = new Layer_t<value_type>(3,64,3,conv1_bin,conv1_bias_bin,argv);
		layers_param.push_back(conv1);
	Layer_t<value_type>* conv2 = new Layer_t<value_type>(64,128,3,conv2_bin,conv2_bias_bin,argv);
		layers_param.push_back(conv2);
	Layer_t<value_type>* conv3 = new Layer_t<value_type>(128,256,3,conv3_bin,conv3_bias_bin,argv);
		layers_param.push_back(conv3);
    Layer_t<value_type>* conv4 = new Layer_t<value_type>(256,256,3,conv4_bin,conv4_bias_bin,argv);
		layers_param.push_back(conv4);
    Layer_t<value_type>* conv5 = new Layer_t<value_type>(256,512,3,conv5_bin,conv5_bias_bin,argv);
		layers_param.push_back(conv5);
    Layer_t<value_type>* conv6 = new Layer_t<value_type>(512,512,3,conv6_bin,conv6_bias_bin,argv);
		layers_param.push_back(conv6);
    Layer_t<value_type>* conv7 = new Layer_t<value_type>(512,512,3,conv7_bin,conv7_bias_bin,argv);
		layers_param.push_back(conv7);
    Layer_t<value_type>* ip = new Layer_t<value_type>(2048,3851,1,ip_bin,ip_bias_bin,argv);
		layers_param.push_back(ip);
	
	int pad1[2] = {1,1};
	int pad0[2] = {0,0};
	int stride1[2] = {1,1};
	int stride2[2] = {2,2};
	int win2[2] = {2,2};
	
	int n,c,h,w;
	n = 1; c = 3; h = IMAGE_H; w = IMAGE_W;
	
	value_type *data = NULL;
	//image
    	imgData_h = new value_type[c*h*w];
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		bottom.push_back( data );
	//conv1
	convs.push_back( new ConvLayer<value_type>(*layers_param[0], n, c, h, w, pad1, stride1) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );
	pools.push_back( new PoolLayer<value_type>(n, c, h, w, win2, pad0, stride2) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );
	//conv2
    convs.push_back( new ConvLayer<value_type>(*layers_param[1], n, c, h, w, pad1, stride1) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );
    pools.push_back( new PoolLayer<value_type>(n, c, h, w, win2, pad0, stride2) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );
	//conv3
	convs.push_back( new ConvLayer<value_type>(*layers_param[2], n, c, h, w, pad1, stride1) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );
	//conv4
	convs.push_back( new ConvLayer<value_type>(*layers_param[3], n, c, h, w, pad1, stride1) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );
    pools.push_back( new PoolLayer<value_type>(n, c, h, w, win2, pad0, stride2) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );
	//conv5
	convs.push_back( new ConvLayer<value_type>(*layers_param[4], n, c, h, w, pad1, stride1) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );
	//bn5
	bns.push_back( new BatchNormLayer<value_type>("bn5_scale.bin","bn5_offset.bin",n,c,h,w,argv) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );
	//conv6
	convs.push_back( new ConvLayer<value_type>(*layers_param[5], n, c, h, w, pad1, stride1) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );
	//bn6	
	bns.push_back( new BatchNormLayer<value_type>("bn6_scale.bin","bn6_offset.bin",n,c,h,w,argv) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );

	pools.push_back( new PoolLayer<value_type>(n, c, h, w, win2, pad0, stride2) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );
    //conv7    
	convs.push_back( new ConvLayer<value_type>(*layers_param[6], n, c, h, w, pad1, stride1) );
		checkCudaErrors( cudaMalloc(&data, n*c*h*w*sizeof(value_type)) );
		top.push_back( data );
		bottom.push_back( data );
	//fc
	fcs.push_back( new FullyConnectedLayer<value_type>(*layers_param[7], n, c, h, h) );
		checkCudaErrors( cudaMalloc(&ipcache, (IMAGE_W/16-1)*n*c*h*h*sizeof(value_type)) );
		checkCudaErrors( cudaMalloc(&logits, (IMAGE_W/16-1)*LOGITNUM*sizeof(value_type)) );
	
	logits_cpu = (value_type*) malloc( (IMAGE_W/16-1)*LOGITNUM*sizeof(value_type) );
	ipcache_cpu = (value_type*) malloc( (IMAGE_W/16-1)*n*c*h*h*sizeof(value_type) );
	topcache_cpu = (value_type*) malloc( (IMAGE_W/16-1)*n*c*h*h*sizeof(value_type));
	//fullyConnectedForward(layers_param[7], n, c, h, w, dstData, &srcData);
    //softmaxForward(n, c, h, w, srcData, &dstData);

    //printDeviceVector(n*c*h*w, dstData);
};

template <typename value_type>
CRNN<value_type>::~CRNN()
{
	//for(int i=0; i<layers_param.size(); i++)
	//	delete layers_param[i];

	for(int i=0; i<fcs.size(); i++)
		delete fcs[i];
	
	for(int i=0; i<bns.size(); i++)
		delete bns[i];

	for(int i=0; i<convs.size(); i++)
		delete convs[i];
	
	for(int i=0; i<pools.size(); i++)
		delete pools[i];
	
	checkCudaErrors( cudaFree(ipcache) );
	checkCudaErrors( cudaFree(logits) );
	free(topcache_cpu);
	free(ipcache_cpu);
	free(logits_cpu);
	checkCudaErrors( cudaFree(bottom[0]) );
	for(int i=0; i<top.size(); i++)
		checkCudaErrors( cudaFree(top[i]) );
}

template <typename value_type>
void CRNN<value_type>::resize(int size, value_type **data)
{
    if (*data != NULL)
    {
        checkCudaErrors( cudaFree(*data) );
	}
    checkCudaErrors( cudaMalloc(data, size*sizeof(value_type)) );
}


template <typename value_type>
std::vector<int> CRNN<value_type>::predict(const void* imgbuf, const int imgw)
{
	if(imgbuf != NULL)
		imgData_h = (value_type*)imgbuf;
	else
		createOnesImage(imgData_h,IMAGE_H,IMAGE_W);

    checkCudaErrors( cudaMemcpy(bottom[0], imgData_h,
                                    3*IMAGE_H*IMAGE_W*sizeof(value_type),
                                    cudaMemcpyHostToDevice) );
    int cnt = 0;
	int poolc = 0;
	int convc = 0;
	int bnc = 0;
	//printDeviceVector(0, 0+512, bottom[cnt]);
	convs[convc]->forward(bottom[cnt],top[cnt]);cnt++;convc++;
	pools[poolc]->forward(bottom[cnt],top[cnt]);cnt++;poolc++;
	
	convs[convc]->forward(bottom[cnt],top[cnt]);cnt++;convc++;
	pools[poolc]->forward(bottom[cnt],top[cnt]);cnt++;poolc++;
	
	convs[convc]->forward(bottom[cnt],top[cnt]);cnt++;convc++;
	//pools[poolc]->forward(bottom[cnt],top[cnt]);cnt++;poolc++;
	
	convs[convc]->forward(bottom[cnt],top[cnt]);cnt++;convc++;
	pools[poolc]->forward(bottom[cnt],top[cnt]);cnt++;poolc++;

	convs[convc]->forward(bottom[cnt],top[cnt]);cnt++;convc++;
	bns[bnc]->forward(bottom[cnt],top[cnt]);cnt++;bnc++;
	
	convs[convc]->forward(bottom[cnt],top[cnt]);cnt++;convc++;
	bns[bnc]->forward(bottom[cnt],top[cnt]);cnt++;bnc++;
	
	pools[poolc]->forward(bottom[cnt],top[cnt]);cnt++;poolc++;	
	
	convs[convc]->forward(bottom[cnt],top[cnt]);
	//printDeviceVector(0,2048, top[cnt]);
	
	int final_channel = 512;
	int final_height = 2;
	int final_width = IMAGE_W/16;
	int perwidth = 2;
	int dimpert = perwidth * final_channel * final_height;
	
	checkCudaErrors( cudaMemcpy(topcache_cpu,top[cnt],
							final_channel*final_height*final_width*sizeof(value_type),cudaMemcpyDeviceToHost) );

	/*
	for(int i=0; i<2048; i++)
		std::cout<<"top_cpu:"<<topcache_cpu[i]<<std::endl;
	*/
	for(int i=0; i<final_width-1; i++)
	{
		for(int h=0; h<final_height; h++)
		{
			for(int c=0; c<final_channel; c++)
			{
				for(int w=0; w<perwidth; w++)
				{
					*(ipcache_cpu + i*dimpert + w*final_height*final_channel + h*final_channel + c) 
					= *(topcache_cpu + (i+w) + c*final_height*final_width + h*final_width); 
					/*
					checkCudaErrors( cudaMemcpy(
							ipcache + i*dimpert + w*final_height*final_channel + h*final_channel + c, 
							top[cnt]+ (i+w) + c*final_height*final_width + h*final_width,
                            sizeof(value_type),
                            cudaMemcpyDeviceToDevice) );
							*/
				}	
			}
		}
		//std::cout<<"t:"<<i<<std::endl;
	}

	for(int i=0; i<final_width-1; i++)
	{
		fcs[0] -> forward_cpu(ipcache_cpu + i*dimpert,logits_cpu+i*LOGITNUM);
		/*
		std::cout<<"fc forward:"<<i<<std::endl;
		for(int k=0; k<10; k++)
		{
			std::cout<<"gemv input:"<<*( ipcache_cpu + i*dimpert + k )<<std::endl;
			std::cout<<"gemv output:"<<*( logits_cpu + i*LOGITNUM + k )<<std::endl;
		}
		*/
	}
	//fcs[0] -> print_param();
    std::vector<int> ids = decode_greedy(logits_cpu,LOGITNUM,imgw/16-1);
	return ids;
}
#endif
