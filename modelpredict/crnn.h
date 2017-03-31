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
#ifndef __CRNN_H__
#define __CRNN_H__

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <cublas_v2.h>

#include "common.h"
#include "Layers.h"

template <typename value_type>
class CRNN
{
public:
    CRNN(std::string binpath, int devnum);
    ~CRNN();
    void resize(int size, value_type **data);
    /*
	void fullyConnectedForward(const Layer_t<value_type>& ip,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData);
    void convoluteForward(const Layer_t<value_type>& conv,
                          int& n, int& c, int& h, int& w,
						  int * padA, int * filterStrideA,
                          value_type* srcData, value_type** dstData);
   
    void poolForward( int& n, int& c, int& h, int& w,
                      int * windowDimA, int * paddingA, int * strideA,
					value_type* srcData, value_type** dstData);
    void softmaxForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData);
    void lrnForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData);
   	void bnForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData);
    void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData);
	*/
	std::vector<int> predict(const void* imgbuf,const int imgw);
private:
	std::vector<Layer_t<value_type>* > layers_param;
	std::vector<ConvLayer<value_type>* > convs;
	std::vector<PoolLayer<value_type>* > pools;
	std::vector<BatchNormLayer<value_type>* > bns;
	std::vector<FullyConnectedLayer<value_type>* > fcs;
	std::vector<value_type* > bottom,top;
	value_type * imgData_h;
	value_type * ipcache, * ipcache_cpu, * topcache_cpu;
	value_type * logits, * logits_cpu;
};

#include "crnn.cpp"
#endif
