#ifndef __COMMON_CPP__
#define __COMMON_CPP__

#include "common.h"
#include <sstream>
#include <fstream>
#include <cblas.h>

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
    sFilename = (std::string("../../bin/") + std::string(fname));
    //sFilename = (std::string("/home/openai/qiaohan/ocr_cudnn/bin/") + std::string(fname));
}

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
void createOnesImage(value_type* imgData_h, int h, int w)
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
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {   
            //int idx = IMAGE_W*i + j;
            //imgData_h[idx] = fromReal(*(oHostSrc.data() + idx) / double(255));
			for (int cc=0; cc<3; cc++)
				imgData_h[3*(w*i + j)+cc] = 1.0;
        }
    } 
}

template <class value_type>
void printDeviceVector(int start, int end, value_type* vec_d)
{
	int size = end;
    typedef value_type real_type;
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
    Convert<real_type> toReal;
    std::cout.precision(7);
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    for (int i = start; i < size; i++)
    {
		//if(i-start==(size-start)/2)
			std::cout<<std::endl;
        std::cout << toReal(vec[i]) << " ";
    }
    std::cout << std::endl;
    delete [] vec;
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
void gemv(cublasHandle_t cublasHandle, int m, int n, double alpha, 
            const double *A, const double *x,
                               double beta, double *y)
{
#ifdef DISABLE_GEMV
    checkCublasErrors( cublasDgemm (cublasHandle, 
                      CUBLAS_OP_T,
                      CUBLAS_OP_N,
                      n,
                      1,
                      m,
                      &alpha, 
                      A, 
                      m,
                      x,
                      m, 
                      &beta, 
                      y,
                      m) );
#else
    checkCublasErrors( cublasDgemv(cublasHandle, CUBLAS_OP_T,
                                  m, n,
                                  &alpha,
                                  A, m,
                                  x, 1,
                                  &beta,
                                  y, 1) );    
#endif
};

void gemv(cublasHandle_t cublasHandle, int m, int n, float alpha, 
            const float *A, const float *x,
                               float beta, float *y)
{
#ifdef DISABLE_GEMV
    checkCublasErrors( cublasSgemm (cublasHandle, 
                      CUBLAS_OP_T,
                      CUBLAS_OP_N,
                      n,
                      1,
                      m,
                      &alpha, 
                      A, 
                      m,
                      x,
                      m, 
                      &beta, 
                      y,
                      m) );
#else
    checkCublasErrors( cublasSgemv(cublasHandle, CUBLAS_OP_T,
                                  m, n,
                                  &alpha,
                                  A, m,
                                  x, 1,
                                  &beta,
                                  y, 1) );    
#endif
};

void gemv_cpu(int m, int n, float alpha, 
            float *A, float *x,
                               float beta, float *y)
{
	/*
	std::cout<<"n:"<<n<<std::endl;
	std::cout<<"m:"<<m<<std::endl;
	for(int i=0; i<10; i++)
	{
		std::cout<<"A:"<<A[i]<<std::endl;
		std::cout<<"x:"<<x[i]<<std::endl;
		std::cout<<"y:"<<y[i]<<std::endl;
	}
	for(int i=0; i<m*n; i++)
		A[i] = 1.0;
	for(int i=0; i<2048; i++)
		x[i] = 1.0;
	*/
	cblas_sgemv(CblasColMajor,CblasTrans,m,n,alpha,A,m,x,1,beta,y,1);	
}

#endif
