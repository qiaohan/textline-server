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

typedef enum {
        FP16_HOST  = 0, 
        FP16_CUDA  = 1,
        FP16_CUDNN = 2
 } fp16Import_t;

// Need the map, since scaling factor is of float type in half precision
// Also when one needs to use float instead of half, e.g. for printing
template <typename T> 
struct ScaleFactorTypeMap { typedef T Type;};


template <typename value_type>
struct Layer_t
{
    fp16Import_t fp16Import;
    int inputs;
    int outputs;
    // linear dimension (i.e. size is kernel_dim * kernel_dim)
    int kernel_dim;
    value_type *data_h, *data_d;
    value_type *bias_h, *bias_d;
    Layer_t(); 
    Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights,
            const char* fname_bias, const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST);
    ~Layer_t();
    
	private:
    void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d);
};


template <typename value_type>
class network_t
{
    typedef typename ScaleFactorTypeMap<value_type>::Type scaling_type;
    int convAlgorithm;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t     poolingDesc;
    cudnnActivationDescriptor_t  activDesc;
    cudnnLRNDescriptor_t   normDesc;
    cublasHandle_t cublasHandle;
	std::vector<Layer_t<value_type>* > layer_params;
	std::vector<value_type*> features;
	void createHandles();
    void destroyHandles();
  public:
    network_t(std::vector<Layer_t<value_type>* > layers);
    ~network_t();
    void resize(int size, value_type **data);
    void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo);
    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t<value_type>& layer, int c, value_type *data);
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
	std::vector<int> classify_example(const char* imgbuf);
	void setup();
};

#include "crnn.cpp"
#endif
