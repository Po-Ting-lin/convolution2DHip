#pragma once
#include <opencv2/opencv.hpp>
#include "fftw3.h"
#include "factorize.h"

void convolution2DOpencv(float* src, float* dst, float* kernel, int iWidth, int iHeight, int kWidth, int kHeight);
void convolution2DNaive(float* src, float* dst, float* kernel, int iWidth, int iHeight, int kWidth, int kHeight);
void convolution2DNaiveMp(float* src, float* dst, float* kernel, int iWidth, int iHeight, int kWidth, int kHeight);
void convolution2DFFTW(float* src, float* dst, float* kernel, int iWidth, int iHeight, int kWidth, int kHeight);
