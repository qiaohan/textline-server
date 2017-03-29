#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <cublas_v2.h>
#include "error_util.h"

typedef enum {
        FP16_HOST  = 0, 
        FP16_CUDA  = 1,
        FP16_CUDNN = 2
} fp16Import_t;

// Need the map, since scaling factor is of float type in half precision
// Also when one needs to use float instead of half, e.g. for printing
/*
template <typename T> 
struct ScaleFactorTypeMap { typedef T Type;};
template <typename value_type> 
typedef typename ScaleFactorTypeMap<value_type>::Type scaling_type;
*/
void get_path(std::string& sFilename, const char *fname, const char *pname);

// float/double <-> half conversion class
template <class value_type>
class Convert
{
	public:
		template <class T>
		value_type operator()(T x) {return value_type(x);}
};

template <class value_type>
void readBinaryFile(const char* fname, int size, value_type* data_h);

template <class value_type>
void readAllocMemcpy(const char* fname, int size, value_type** data_h, value_type** data_d);

template <class value_type>
void createOnesImage(value_type* imgData_h);

template <class value_type>
void printDeviceVector(int start, int end, value_type* vec_d);

void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                    cudnnTensorFormat_t& tensorFormat,
                    cudnnDataType_t& dataType,
                    int n,
                    int c,
                    int h,
                    int w);
void gemv(cublasHandle_t cublasHandle, int m, int n, double alpha, 
            const double *A, const double *x,
                               double beta, double *y);
void gemv(cublasHandle_t cublasHandle, int m, int n, float alpha, 
            const float *A, const float *x,
                               float beta, float *y);		
void gemv_cpu(int m, int n, float alpha, 
            float *A, float *x,
                               float beta, float *y);		
#include "common.cpp"
#endif
