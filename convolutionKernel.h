#pragma once
#include "hip/hip_runtime.h"

#define WRAP_SIZE 32
#define BLOCK_DIM 32

static inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);    
}

static void gpuWarmUp(){
    hipFree(0);
}

static void gpuReset(){
    hipDeviceReset();
}

void convolution2DNaiveHip(float* src, float* dst, float* kernel, int iw, int ih, int kw, int kh);
void convolution2DSepHip(float* src, float* dst, float* rowKernel, float* colKernel, int iw, int ih, int kw, int kh);
void convolution2DSepConstHip(float* src, float* dst, float* rowKernel, float* colKernel, int iw, int ih, int kw, int kh);

__global__ void convolution2DNaiveHipKernel(float* dSrc, float* dDst, float* dKernel, int iw, int ih, int kw, int kh);
__global__ void convolution2DSepColHipKernel(float* dSrc, float* dDst, float* dKernel, int iw, int ih, int kh);
__global__ void convolution2DSepRowHipKernel(float* dSrc, float* dDst, float* dKernel, int iw, int ih, int kw);
