#ifndef __LAYERS_CPP__
#define __LAYERS_CPP__

#include "Layers.h"

template <typename T>
Layer_t<T>::~Layer_t()
{
	if (data_h != NULL) delete [] data_h;
    if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
    if (bias_h != NULL) delete [] bias_h;
    if (bias_d != NULL) checkCudaErrors( cudaFree(bias_d) );
	data_h = data_d = bias_h = bias_d = NULL;
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

template <typename value_type>
ConvLayer<value_type>::ConvLayer(Layer_t<value_type>& conv, int& n, int& c, int& h, int& w, int * padA, int * filterStrideA)
{
	layer_param = conv;
	switch (sizeof(value_type))
    {
        case 2 : dataType = CUDNN_DATA_HALF; break;
        case 4 : dataType = CUDNN_DATA_FLOAT; break;
        case 8 : dataType = CUDNN_DATA_DOUBLE; break;
        default : FatalError("Unsupported data type");
    }
    tensorFormat = CUDNN_TENSOR_NCHW;
    createHandles();
	
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
												//CUDNN_CONVOLUTION,
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


	// Choose the best according to the preference
    /*
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
	*/
	algo = cudnnConvolutionFwdAlgo_t(0);
	// workspaces
    checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                algo,
                                                &sizeInBytes) );
    
	if(sizeInBytes!=0)
	{
		checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
	}
	else
	{
		std::cout<<"workspace:"<<sizeInBytes<<std::endl;
	}
	setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);
}

template <typename value_type>
ConvLayer<value_type>::~ConvLayer()
{
	if (sizeInBytes!=0)
    {
		checkCudaErrors( cudaFree(workSpace) );
    }
	destroyHandles();
}

template <typename value_type>
void ConvLayer<value_type>::createHandles()
{
	checkCUDNN( cudnnCreate(&cudnnHandle) );
    checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
	checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
    checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
}
template <typename value_type>
void ConvLayer<value_type>::destroyHandles()
{
    checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
	checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
    checkCUDNN( cudnnDestroy(cudnnHandle) );
}

template <typename value_type>
void ConvLayer<value_type>::forward( value_type* srcData, value_type* dstData)
{
    checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                        &alpha_w,
                                        srcTensorDesc,
                                        srcData,
                                        filterDesc,
                                        layer_param.data_d,
                                        convDesc,
                                        algo,
                                        workSpace,
										sizeInBytes,
                                        &beta_w,
                                        dstTensorDesc,
                                        dstData) );   

    checkCUDNN( cudnnAddTensor( cudnnHandle, 
                                &alpha_b, 
								biasTensorDesc,
                                layer_param.bias_d,
                                &beta_b,
                                dstTensorDesc,
                                dstData) );	
}

template <typename value_type>
FullyConnectedLayer<value_type>::FullyConnectedLayer(Layer_t<value_type>& lp, int& n, int& c, int& h, int& w)
{
	ip = lp;
	if (n != 1)
    {
        FatalError("Not Implemented"); 
    }
    dim_x = c*h*w;
    dim_y = ip.outputs;
	h = 1; w = 1; c = dim_y;
	checkCublasErrors( cublasCreate(&cublasHandle) );
}

template <typename value_type>
FullyConnectedLayer<value_type>::~FullyConnectedLayer()
{
	checkCublasErrors( cublasDestroy(cublasHandle) );
}

template <typename value_type>
void FullyConnectedLayer<value_type>::print_param()
{
	for(int i=0; i<10; i++)
	{
		std::cout<<"weights:"<<ip.data_h[i]<<std::endl;
		std::cout<<"bias:"<<ip.bias_h[i]<<std::endl;
	}
}

template <typename value_type>
void FullyConnectedLayer<value_type>::forward_cpu(value_type* srcData, value_type* dstData)
{
    // place bias into dstData
    memcpy(dstData, ip.bias_h, dim_y*sizeof(value_type));
    gemv_cpu(dim_x, dim_y, alpha, ip.data_h, srcData, beta, dstData);
}

template <typename value_type>
void FullyConnectedLayer<value_type>::forward(value_type* srcData, value_type* dstData)
{
    // place bias into dstData
    checkCudaErrors( cudaMemcpy(dstData, ip.bias_d, dim_y*sizeof(value_type), cudaMemcpyDeviceToDevice) );
    gemv(cublasHandle, dim_x, dim_y, alpha, ip.data_d, srcData, beta, dstData);
	//cublasSgemv(cublasHandle,CUBLAS_OP_N, dim_x, dim_y, &alpha, ip.data_d, dim_x, srcData, 1, &beta, dstData, 1);
}

template <typename value_type>
PoolLayer<value_type>::PoolLayer(int& n, int& c, int& h, int& w, int * windowDimA, int * paddingA, int * strideA)
{
	switch (sizeof(value_type))
    {
        case 2 : dataType = CUDNN_DATA_HALF; break;
        case 4 : dataType = CUDNN_DATA_FLOAT; break;
        case 8 : dataType = CUDNN_DATA_DOUBLE; break;
        default : FatalError("Unsupported data type");
    }
    tensorFormat = CUDNN_TENSOR_NCHW;
    createHandles();
	
    setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w); 	
	const int poolDims = 2;
	checkCUDNN( cudnnSetPoolingNdDescriptor(poolingDesc,
                                            CUDNN_POOLING_MAX,
                                            CUDNN_PROPAGATE_NAN,
                                            poolDims,
                                            windowDimA,
                                            paddingA,
                                            strideA ) );
											
	const int tensorDims = 4;
    int tensorOuputDimA[tensorDims] = {n,c,h,w};
    checkCUDNN( cudnnGetPoolingNdForwardOutputDim(poolingDesc,
                                                  srcTensorDesc,
                                                  tensorDims,
                                                  tensorOuputDimA) );
    n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
    h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

    setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);
}

template <typename value_type>
PoolLayer<value_type>::~PoolLayer()
{
	destroyHandles();
}
template <typename value_type>
void PoolLayer<value_type>::createHandles()
{
    checkCUDNN( cudnnCreate(&cudnnHandle) );
    checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
    checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc) );
}
template <typename value_type>
void PoolLayer<value_type>::destroyHandles()
{
    checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
    checkCUDNN( cudnnDestroy(cudnnHandle) );
}
template <typename value_type>
void PoolLayer<value_type>::forward( value_type* srcData, value_type* dstData)
{
	checkCUDNN( cudnnPoolingForward(cudnnHandle,
                                    poolingDesc,
                                    &alpha,
                                    srcTensorDesc,
                                    srcData,
                                    &beta,
                                    dstTensorDesc,
									dstData) );
}

template <typename value_type>
BatchNormLayer<value_type>::BatchNormLayer(std::string scalefname, std::string offsetfname, int& n, int& c, int& h, int& w)
{
	switch (sizeof(value_type))
    {
        case 2 : dataType = CUDNN_DATA_HALF; break;
        case 4 : dataType = CUDNN_DATA_FLOAT; break;
        case 8 : dataType = CUDNN_DATA_DOUBLE; break;
        default : FatalError("Unsupported data type");
    }
    tensorFormat = CUDNN_TENSOR_NCHW;
    createHandles();
	
    setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w); 	
    setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);
	//setTensorDesc(bnScaleBiasMeanVarDesc, tensorFormat, dataType, 1, c, 1, 1);
	checkCUDNN( cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc,srcTensorDesc,mode) );
	loadparams(scalefname, offsetfname, c);
}

template <typename value_type>
void BatchNormLayer<value_type>::createHandles(){
	checkCUDNN( cudnnCreate(&cudnnHandle) );
    checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc) );
}

template <typename value_type>
void BatchNormLayer<value_type>::destroyHandles(){
    checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc) );
    checkCUDNN( cudnnDestroy(cudnnHandle) );
}

template <typename value_type>
void BatchNormLayer<value_type>::loadparams(std::string scalefname, std::string offsetfname, int channel){
	
	std::string weights_path,bias_path;
    get_path(weights_path, scalefname.c_str(), "bn");
    get_path(bias_path, offsetfname.c_str(), "bn");
	
	checkCudaErrors( cudaMalloc(&mean, channel) );
	checkCudaErrors( cudaMalloc(&var, channel) );
    readAllocMemcpy<value_type>(weights_path.c_str(), channel, &bnScale_h, &bnScale_d);
    readAllocMemcpy<value_type>(bias_path.c_str(), channel, &bnBias_h, &bnBias_d);
}

template <typename value_type>
BatchNormLayer<value_type>::~BatchNormLayer()
{
	if (bnScale_h != NULL) delete [] bnScale_h;
    if (bnScale_d != NULL) checkCudaErrors( cudaFree(bnScale_d) );
    if (bnBias_h != NULL) delete [] bnBias_h;
    if (bnBias_d != NULL) checkCudaErrors( cudaFree(bnBias_d) );
	
	if (mean != NULL) checkCudaErrors( cudaFree(mean) );
	if (var != NULL) checkCudaErrors( cudaFree(var) );
	mean = var = bnScale_h = bnScale_d = bnBias_h = bnBias_d = NULL;
	destroyHandles();
}
		
template <typename value_type>
void BatchNormLayer<value_type>::forward(value_type* srcData, value_type* dstData)
{
	checkCUDNN( cudnnBatchNormalizationForwardTraining(cudnnHandle,
									mode,
                                    &alpha,
                                    &beta,
									srcTensorDesc,
                                    srcData,
                                    dstTensorDesc,
									dstData,
									bnScaleBiasMeanVarDesc,
									bnScale_d,
									bnBias_d,
									1.0,
									mean,
									var,
									1e-3,
									NULL,
									NULL) );
	/*
	value_type *tt, *t2, *t1;
	checkCudaErrors( cudaMalloc(&tt, 4*4*512) );
	checkCudaErrors( cudaMalloc(&t2, 4*4*512) );
	checkCudaErrors( cudaMalloc(&t1, 512) );
	checkCUDNN( cudnnBatchNormalizationForwardInference(cudnnHandle,
                                    mode,
                                    &alpha,
                                    &beta,
									srcTensorDesc,
                                    srcData,
									//tt,
                                    dstTensorDesc,
									dstData,
									bnScaleBiasMeanVarDesc,
									bnScale_d,
									bnBias_d,
									mean,
									var,
									1e-3) );
									*/
	//value_type * t = new value_type[4*4*512];
	//cudaMemcpy( dstData, t, 4*4*512, cudaMemcpyHostToDevice);
	//delete t;
}
#endif
