#ifndef __LAYERS_H__
#define __LAYERS_H__

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <cublas_v2.h>
#include "common.h"
#include <string>

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
class ConvLayer
{
	public:
		ConvLayer(Layer_t<value_type>& conv, int& n, int& c, int& h, int& w, int * padA, int * filterStrideA);
		~ConvLayer();
		void forward(value_type* srcData, value_type* dstData);
	private:
		void createHandles();
		void destroyHandles();
	
		cudnnConvolutionDescriptor_t convDesc;
		cudnnFilterDescriptor_t filterDesc;
		cudnnTensorDescriptor_t biasTensorDesc,dstTensorDesc,srcTensorDesc;
		cudnnConvolutionFwdAlgo_t algo;
		cudnnHandle_t cudnnHandle;
		
		Layer_t<value_type> layer_param;
		
		size_t sizeInBytes=0;
		void* workSpace=NULL;
		const float alpha_w = float(1);
        const float beta_w  = float(0);
		const float alpha_b = float(1);
        const float beta_b  = float(1);
		cudnnDataType_t dataType;
		cudnnTensorFormat_t tensorFormat;
};

template <typename value_type>
class FullyConnectedLayer
{
	public:
		FullyConnectedLayer(Layer_t<value_type>& lp, int& n, int& c, int& h, int& w);
		~FullyConnectedLayer();
		void forward(value_type* srcData, value_type* dstData);
		void forward_cpu(value_type* srcData, value_type* dstData);
		void print_param();
	private:	
		cublasHandle_t cublasHandle;
		int dim_x,dim_y;		
		float alpha = float(1), beta = float(1);
		Layer_t<value_type> ip;
};

template <typename value_type>
class BatchNormLayer
{
	public:
		BatchNormLayer(std::string scalefname, std::string offsetfname, int& n, int& c, int& h, int& w);
		~BatchNormLayer();
		void forward(value_type* srcData, value_type* dstData);
	private:
		void createHandles();
		void destroyHandles();
		void loadparams(std::string scalefname, std::string offsetfname, int channel);
	
		cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,dstTensorDesc,srcTensorDesc;
		cudnnHandle_t cudnnHandle;
		cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
		
		const float alpha = float(1);
        const float beta  = float(0);
		
		cudnnDataType_t dataType;
		cudnnTensorFormat_t tensorFormat;
		value_type * mean;
		value_type * var;
		value_type * bnScale_d;
		value_type * bnBias_d;
		value_type * bnScale_h;
		value_type * bnBias_h;
};

template <typename value_type>
class PoolLayer
{
	public:
		PoolLayer(int& n, int& c, int& h, int& w, int * windowDimA, int * paddingA, int * strideA);
		~PoolLayer();
		void forward(value_type* srcData, value_type* dstData);
	private:
		void createHandles();
		void destroyHandles();
	
		cudnnPoolingDescriptor_t poolingDesc;
		cudnnTensorDescriptor_t dstTensorDesc,srcTensorDesc;
		cudnnHandle_t cudnnHandle;
		
		size_t sizeInBytes=0;
		void* workSpace=NULL;
		const float alpha = float(1);
        const float beta = float(0);
		cudnnDataType_t dataType;
		cudnnTensorFormat_t tensorFormat;
};

#include "Layers.cpp"
#endif
