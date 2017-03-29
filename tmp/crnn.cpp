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

#define IMAGE_H 32
#define IMAGE_W 32



/********************************************************
 * Prints the error message, and exits
 * ******************************************************/

#define EXIT_WAIVED 0
	
template <typename T>
Layer_t<T>::~Layer_t()
{
	if (data_h != NULL) delete [] data_h;
    if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
    if (bias_h != NULL) delete [] bias_h;
    if (bias_d != NULL) checkCudaErrors( cudaFree(bias_d) );
}


void get_path(std::string& sFilename, const char *fname, const char *pname)
{
    sFilename = (std::string("../bin/") + std::string(fname));
}


// float/double <-> half conversion class
template <class value_type>
class Convert
{
public:
    template <class T>
    value_type operator()(T x) {return value_type(x);}
};


// IO utils
template <class value_type>
void readBinaryFile(const char* fname, int size, value_type* data_h)
{
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    std::stringstream error_s;
    if (!dataFile)
    {
        error_s << "Error opening file " << fname; 
        FatalError(error_s.str());
    }
    // we assume the data stored is always in float precision
    float* data_tmp = new float[size];
    int size_b = size*sizeof(float);
	std::cout<< "size(Bytes): "<< size_b<< std::endl;
    if (!dataFile.read ((char*) data_tmp, size_b)) 
    {
        error_s << "Error reading file " << fname ; 
        FatalError(error_s.str());
    }
    // conversion
    Convert<value_type> fromReal;
    for (int i = 0; i < size; i++)
    {
        data_h[i] = fromReal(data_tmp[i]);
    }
    delete [] data_tmp;
}

template <class value_type>
void readAllocMemcpy(const char* fname, int size, value_type** data_h, value_type** data_d)
{
    *data_h = new value_type[size];

    readBinaryFile<value_type>(fname, size, *data_h);

    int size_b = size*sizeof(value_type);
    checkCudaErrors( cudaMalloc(data_d, size_b) );
    checkCudaErrors( cudaMemcpy(*data_d, *data_h,
                                size_b,
                                cudaMemcpyHostToDevice) );
}

template <class value_type>
void createOnesImage(value_type* imgData_h)
{
    /*
	// declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    std::string sFilename(fname);
    std::cout << "Loading image " << sFilename << std::endl;
    // Take care of half precision
    Convert<value_type> fromReal;
    // load gray-scale image from disk
    try
    {
        npp::loadImage(sFilename, oHostSrc);
    }
    catch (npp::Exception &rException)
    {
        FatalError(rException.toString());
    }
    */
	// Plot to console and normalize image to be in range [0,1]
    for (int i = 0; i < IMAGE_H; i++)
    {
        for (int j = 0; j < IMAGE_W; j++)
        {   
            //int idx = IMAGE_W*i + j;
            //imgData_h[idx] = fromReal(*(oHostSrc.data() + idx) / double(255));
			for (int cc=0; cc<3; cc++)
				imgData_h[3*(IMAGE_W*i + j)+cc] = 1.0;
        }
    } 
}

template <class value_type>
void printDeviceVector(int start, int end, value_type* vec_d)
{
	int size = end;
    typedef typename ScaleFactorTypeMap<value_type>::Type real_type;
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
    Convert<real_type> toReal;
    std::cout.precision(7);
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    for (int i = start; i < size; i++)
    {
		if(i-start==(size-start)/2)
			std::cout<<std::endl;
        std::cout << toReal(vec[i]) << " ";
    }
    std::cout << std::endl;
    delete [] vec;
}

template <typename value_type>
Layer_t<value_type>::Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), kernel_dim(0), fp16Import(FP16_HOST){};
template <typename value_type>
	Layer_t<value_type>::Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights,
            const char* fname_bias, const char* pname, fp16Import_t _fp16Import)
                  : inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim)
    {
        fp16Import = _fp16Import;
        std::string weights_path, bias_path;
        if (pname != NULL)
        {
            get_path(weights_path, fname_weights, pname);
            get_path(bias_path, fname_bias, pname);
        }
        else
        {
            weights_path = fname_weights; bias_path = fname_bias;
        }
        readAllocInit(weights_path.c_str(), inputs * outputs * kernel_dim * kernel_dim, 
                        &data_h, &data_d);
        readAllocInit(bias_path.c_str(), outputs, &bias_h, &bias_d);
    }
	

template <typename value_type>
	void Layer_t<value_type>::readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
    {
        readAllocMemcpy<value_type>(fname, size, data_h, data_d);
    }


// demonstrate different ways of setting tensor descriptor
//#define SIMPLE_TENSOR_DESCRIPTOR
#define ND_TENSOR_DESCRIPTOR
void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                    cudnnTensorFormat_t& tensorFormat,
                    cudnnDataType_t& dataType,
                    int n,
                    int c,
                    int h,
                    int w)
{
#if SIMPLE_TENSOR_DESCRIPTOR
    checkCUDNN( cudnnSetTensor4dDescriptor(tensorDesc,
                                            tensorFormat,
                                            dataType,
                                            n, c,
                                            h,
                                            w ) );
#elif defined(ND_TENSOR_DESCRIPTOR)
    const int nDims = 4;
    int dimA[nDims] = {n,c,h,w};
    int strideA[nDims] = {c*h*w, h*w, w, 1};
    checkCUDNN( cudnnSetTensorNdDescriptor(tensorDesc,
                                            dataType,
                                            4,
                                            dimA,
                                            strideA ) ); 
#else
    checkCUDNN( cudnnSetTensor4dDescriptorEx(tensorDesc,
                                            dataType,
                                            n, c,
                                            h, w,
                                            c*h*w, h*w, w, 1) );
#endif
}

template <typename value_type>
    void network_t<value_type>::createHandles()
    {
        checkCUDNN( cudnnCreate(&cudnnHandle) );
        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
        checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
        checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
        checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc) );
        checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
        checkCUDNN( cudnnCreateLRNDescriptor(&normDesc) );

        checkCublasErrors( cublasCreate(&cublasHandle) );
    }
template <typename value_type>
    void network_t<value_type>::destroyHandles()
    {
        checkCUDNN( cudnnDestroyLRNDescriptor(normDesc) );
        checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc) );
        checkCUDNN( cudnnDestroyActivationDescriptor(activDesc) );
        checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
        checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
        checkCUDNN( cudnnDestroy(cudnnHandle) );

        checkCublasErrors( cublasDestroy(cublasHandle) );
    }
template <typename value_type>
    network_t<value_type>::network_t(std::vector<Layer_t<value_type>* > layers)
    {
		layer_params = layers;
        convAlgorithm = -1;
        switch (sizeof(value_type))
        {
            case 2 : dataType = CUDNN_DATA_HALF; break;
            case 4 : dataType = CUDNN_DATA_FLOAT; break;
            case 8 : dataType = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
        tensorFormat = CUDNN_TENSOR_NCHW;
        createHandles();    
		checkCudaErrors( cudaSetDevice(0) );
    };
template <typename value_type>
    network_t<value_type>::~network_t()
    {
        destroyHandles();
    }
template <typename value_type>
    void network_t<value_type>::resize(int size, value_type **data)
    {
        if (*data != NULL)
        {
            checkCudaErrors( cudaFree(*data) );
        }
        checkCudaErrors( cudaMalloc(data, size*sizeof(value_type)) );
    }
template <typename value_type>
    void network_t<value_type>::setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
    {
        convAlgorithm = (int) algo;
    }
template <typename value_type>
    void network_t<value_type>::addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t<value_type>& layer, int c, value_type *data)
    {
        setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(1);
        checkCUDNN( cudnnAddTensor( cudnnHandle, 
                                    &alpha, biasTensorDesc,
                                    layer.bias_d,
                                    &beta,
                                    dstTensorDesc,
                                    data) );
    }
template <typename value_type>
    void network_t<value_type>::fullyConnectedForward(const Layer_t<value_type>& ip,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        if (n != 1)
        {
            FatalError("Not Implemented"); 
        }
        int dim_x = c*h*w;
        int dim_y = ip.outputs;
        resize(dim_y, dstData);

        scaling_type alpha = scaling_type(1), beta = scaling_type(1);
        // place bias into dstData
        checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, dim_y*sizeof(value_type), cudaMemcpyDeviceToDevice) );
        
        gemv(cublasHandle, dim_x, dim_y, alpha,
                ip.data_d, srcData, beta,*dstData);

        h = 1; w = 1; c = dim_y;
    }
template <typename value_type>
    void network_t<value_type>::convoluteForward(const Layer_t<value_type>& conv,
                          int& n, int& c, int& h, int& w,
						  int * padA, int * filterStrideA,
                          value_type* srcData, value_type** dstData)
    {
        cudnnConvolutionFwdAlgo_t algo;

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        const int filterDimA[tensorDims] = {conv.outputs, conv.inputs, 
                                        conv.kernel_dim, conv.kernel_dim};
                                       
        checkCUDNN( cudnnSetFilterNdDescriptor(filterDesc,
                                              dataType,
                                              CUDNN_TENSOR_NCHW,
                                              tensorDims,
                                              filterDimA) );
 
        const int convDims = 2;
        //int padA[convDims] = {0,0};
        //int filterStrideA[convDims] = {1,1};
        int upscaleA[convDims] = {1,1};
        cudnnDataType_t  convDataType = dataType;
        if (dataType == CUDNN_DATA_HALF) {
            convDataType = CUDNN_DATA_FLOAT; //Math are done in FP32 when tensor are in FP16
        }
        checkCUDNN( cudnnSetConvolutionNdDescriptor(convDesc,
                                                    convDims,
                                                    padA,
                                                    filterStrideA,
                                                    upscaleA,
                                                    CUDNN_CROSS_CORRELATION,
                                                    convDataType) );
        // find dimension of convolution output
        checkCUDNN( cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                                srcTensorDesc,
                                                filterDesc,
                                                tensorDims,
                                                tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        if (convAlgorithm < 0)
        {
            // Choose the best according to the preference
            std::cout << "Testing cudnnGetConvolutionForwardAlgorithm ...\n";
            checkCUDNN( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                    srcTensorDesc,
                                                    filterDesc,
                                                    convDesc,
                                                    dstTensorDesc,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    0,
                                                    &algo
                                                    ) );
            std::cout << "Fastest algorithm is Algo " << algo << "\n";
            convAlgorithm = algo;
            // New way of finding the fastest config
            // Setup for findFastest call
            std::cout << "Testing cudnnFindConvolutionForwardAlgorithm ...\n";
            int requestedAlgoCount = 5; 
            int returnedAlgoCount[1];
            cudnnConvolutionFwdAlgoPerf_t *results = (cudnnConvolutionFwdAlgoPerf_t*)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t)*requestedAlgoCount);
            checkCUDNN(cudnnFindConvolutionForwardAlgorithm( cudnnHandle, 
                                                     srcTensorDesc,
                                                     filterDesc,
                                                     convDesc,
                                                     dstTensorDesc,
                                                     requestedAlgoCount,
                                                     returnedAlgoCount,
                                                     results
                                                   ) );
        for(int algoIndex = 0; algoIndex < *returnedAlgoCount; ++algoIndex){
            printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", cudnnGetErrorString(results[algoIndex].status), results[algoIndex].algo, results[algoIndex].time, (unsigned long long)results[algoIndex].memory);
        }
            free(results);
        }
        else
        {
            algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;
            if (algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
            {
                //std::cout << "Using FFT for convolution\n";
            }
        }

        resize(n*c*h*w, dstData);
        size_t sizeInBytes=0;
        void* workSpace=NULL;
        checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                algo,
                                                &sizeInBytes) );
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
        }
        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                              &alpha,
                                              srcTensorDesc,
                                              srcData,
                                              filterDesc,
                                              conv.data_d,
                                              convDesc,
                                              algo,
                                              workSpace,
                                              sizeInBytes,
                                              &beta,
                                              dstTensorDesc,
                                              *dstData) );
        addBias(dstTensorDesc, conv, c, *dstData);
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaFree(workSpace) );
        }
    }

template <typename value_type>
    void network_t<value_type>::poolForward( int& n, int& c, int& h, int& w,
                      int * windowDimA, int * paddingA, int * strideA,
					value_type* srcData, value_type** dstData)
    {
		const int poolDims = 2;
        /*
        int windowDimA[poolDims] = {2,2};
        int paddingA[poolDims] = {0,0};
        int strideA[poolDims] = {2,2};
        */
		checkCUDNN( cudnnSetPoolingNdDescriptor(poolingDesc,
                                                CUDNN_POOLING_MAX,
                                                CUDNN_PROPAGATE_NAN,
                                                poolDims,
                                                windowDimA,
                                                paddingA,
                                                strideA ) );

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);        

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        checkCUDNN( cudnnGetPoolingNdForwardOutputDim(poolingDesc,
                                                    srcTensorDesc,
                                                    tensorDims,
                                                    tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);  
     
        resize(n*c*h*w, dstData);
        scaling_type alpha = scaling_type(1);
        scaling_type beta = scaling_type(0);
        checkCUDNN( cudnnPoolingForward(cudnnHandle,
                                          poolingDesc,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }
template <typename value_type>
    void network_t<value_type>::softmaxForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
                                          CUDNN_SOFTMAX_ACCURATE ,
                                          CUDNN_SOFTMAX_MODE_CHANNEL,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }
template <typename value_type>
    void network_t<value_type>::lrnForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        unsigned lrnN = 5;
        double lrnAlpha, lrnBeta, lrnK;
        lrnAlpha = 0.0001; lrnBeta = 0.75; lrnK = 1.0;
        checkCUDNN( cudnnSetLRNDescriptor(normDesc,
                                            lrnN,
                                            lrnAlpha,
                                            lrnBeta,
                                            lrnK) );

        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnLRNCrossChannelForward(cudnnHandle,
                                            normDesc,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );
    }
template <typename value_type>
	void network_t<value_type>::bnForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        unsigned lrnN = 5;
        double lrnAlpha, lrnBeta, lrnK;
        lrnAlpha = 0.0001; lrnBeta = 0.75; lrnK = 1.0;
        checkCUDNN( cudnnSetLRNDescriptor(normDesc,
                                            lrnN,
                                            lrnAlpha,
                                            lrnBeta,
                                            lrnK) );

        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnBatchNormalizationForwardInference(cudnnHandle,
                                            CUDNN_BATCHNORM_SPATIAL,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );
    }
template <typename value_type>
    void network_t<value_type>::activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                                CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN,
                                                0.0) );
    
        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnActivationForward(cudnnHandle,
                                            activDesc,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );    
    }

template <typename value_type>
void network_t<value_type>::setup(const char* imgbuf)
{
	features.clear();
    int n,c,h,w;
    value_type *srcData = NULL, *dstData = NULL;
    value_type *imgData_h = new value_type[3*IMAGE_H*IMAGE_W];

    createOnesImage(imgData_h);


    checkCudaErrors( cudaMalloc(&srcData, 3*IMAGE_H*IMAGE_W*sizeof(value_type)) );
    checkCudaErrors( cudaMemcpy(srcData, imgData_h,
                                    3*IMAGE_H*IMAGE_W*sizeof(value_type),
                                    cudaMemcpyHostToDevice) );
    features.push_back(srcData);
	n = 1; c = 3; h = IMAGE_H; w = IMAGE_W;
       
	int pad1[2] = {1,1};
	int pad0[2] = {0,0};
	int stride1[2] = {1,1};
	int stride2[2] = {2,2};
	int win2[2] = {2,2};
	
	convoluteForward(*layer_params[0], n, c, h, w, pad1, stride1, srcData, &dstData);
        poolForward(n, c, h, w, win2, pad0, stride2, dstData, &srcData);
    convoluteForward(*layer_params[1], n, c, h, w, pad1, stride1, srcData, &dstData);
        poolForward(n, c, h, w, win2, pad0, stride2, dstData, &srcData);
		
	convoluteForward(*layer_params[2], n, c, h, w, pad1, stride1, srcData, &dstData);
        //poolForward(n, c, h, w, dstData, &srcData);
 
	convoluteForward(*layer_params[3], n, c, h, w, pad1, stride1, dstData, &srcData);
        poolForward(n, c, h, w, win2, pad0, stride2, srcData, &dstData);     

	convoluteForward(*layer_params[4], n, c, h, w, pad1, stride1, dstData, &srcData);
		//bn5
		//printDeviceVector(0,8, srcData);

	convoluteForward(*layer_params[5], n, c, h, w, pad1, stride1, srcData, &dstData);
        //bn6
		poolForward(n, c, h, w, win2, pad0, stride2, dstData, &srcData);
        
	convoluteForward(*layer_params[6], n, c, h, w, pad1, stride1, srcData, &dstData);
		
		//fullyConnectedForward(layer_params[7], n, c, h, w, dstData, &srcData);
        //softmaxForward(n, c, h, w, srcData, &dstData);

        //printDeviceVector(n*c*h*w, dstData);
		std::vector<int> id;
		/*
        const int max_digits = 3851;
        // Take care of half precision
        Convert<scaling_type> toReal;
        value_type result[max_digits];
        checkCudaErrors( cudaMemcpy(result, dstData, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
        for (int i = 1; i < max_digits; i++)
        {
            if (toReal(result[id]) < toReal(result[i])) id = i;
        }

        std::cout << "Resulting weights from Softmax:" << std::endl;
		*/
        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
        return id;
}
template <typename value_type>
	std::vector<int> network_t<value_type>::classify_example(const char* imgbuf)
    {
        int n,c,h,w;
        value_type *srcData = NULL, *dstData = NULL;
        value_type imgData_h[3*IMAGE_H*IMAGE_W];

        createOnesImage(imgData_h);

        std::cout << "Performing forward propagation ...\n";

        checkCudaErrors( cudaMalloc(&srcData, 3*IMAGE_H*IMAGE_W*sizeof(value_type)) );
        checkCudaErrors( cudaMemcpy(srcData, imgData_h,
                                    3*IMAGE_H*IMAGE_W*sizeof(value_type),
                                    cudaMemcpyHostToDevice) );

        n = 1; c = 3; h = IMAGE_H; w = IMAGE_W;
       
		int pad1[2] = {1,1};
		int pad0[2] = {0,0};
		int stride1[2] = {1,1};
		int stride2[2] = {2,2};
		int win2[2] = {2,2};
		convoluteForward(*layer_params[0], n, c, h, w, pad1, stride1, srcData, &dstData);
        poolForward(n, c, h, w, win2, pad0, stride2, dstData, &srcData);
        convoluteForward(*layer_params[1], n, c, h, w, pad1, stride1, srcData, &dstData);
        poolForward(n, c, h, w, win2, pad0, stride2, dstData, &srcData);
		
		convoluteForward(*layer_params[2], n, c, h, w, pad1, stride1, srcData, &dstData);
        //poolForward(n, c, h, w, dstData, &srcData);
 
		convoluteForward(*layer_params[3], n, c, h, w, pad1, stride1, dstData, &srcData);
        poolForward(n, c, h, w, win2, pad0, stride2, srcData, &dstData);     

		convoluteForward(*layer_params[4], n, c, h, w, pad1, stride1, dstData, &srcData);
		//bn5
		printDeviceVector(0,8, srcData);

		convoluteForward(*layer_params[5], n, c, h, w, pad1, stride1, srcData, &dstData);
        //bn6
		poolForward(n, c, h, w, win2, pad0, stride2, dstData, &srcData);
        
		convoluteForward(*layer_params[6], n, c, h, w, pad1, stride1, srcData, &dstData);
		
		//fullyConnectedForward(layer_params[7], n, c, h, w, dstData, &srcData);
        //softmaxForward(n, c, h, w, srcData, &dstData);

        //printDeviceVector(n*c*h*w, dstData);
		std::vector<int> id;
		/*
        const int max_digits = 3851;
        // Take care of half precision
        Convert<scaling_type> toReal;
        value_type result[max_digits];
        checkCudaErrors( cudaMemcpy(result, dstData, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
        for (int i = 1; i < max_digits; i++)
        {
            if (toReal(result[id]) < toReal(result[i])) id = i;
        }

        std::cout << "Resulting weights from Softmax:" << std::endl;
		*/
        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
        return id;
    }
