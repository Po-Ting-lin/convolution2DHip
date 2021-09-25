#include "convolution.h"
#include "convolutionKernel.h"
#include "timer.h"
#include "utils.h"

template<typename F>
void testFunction(const char* name, int repeatTime, F func, float* src, float* dst, float* kernel, int iWidth, int iHeight, int kWidth, int kHeight) {
    printf("%s: %.4f ms (average %d times)\n", name, funcTime(repeatTime, func, src, dst, kernel, iWidth, iHeight, kWidth, kHeight), repeatTime);
    printf("Correctness: %s\n\n", checkCorrectness(src, dst, kernel, iWidth, iHeight, kWidth, kHeight));	
}

int main() {
    gpuWarmUp();
    const int width = 1000;
    const int height = 200;
    const int kernel_width = 31;
    const int kernel_height = 31;

    float* kernel = new float[kernel_width * kernel_height];
    float* src_image = new float[width * height];
    float* dst_image = new float[width * height];
    getRandomImage(src_image, width, height);
    getGaussianKernel2D(kernel, kernel_width, kernel_height);

    const int repeat_times = 10;

    testFunction("opencv convolution", repeat_times, convolution2DOpencv, src_image, dst_image, kernel, width, height, kernel_width, kernel_height);   
    testFunction("naive convolution", 1, convolution2DNaive, src_image, dst_image, kernel, width, height, kernel_width, kernel_height);
    testFunction("navie MP convolution", repeat_times, convolution2DNaiveMp, src_image, dst_image, kernel, width, height, kernel_width, kernel_height);
    testFunction("navie GPU convolution", repeat_times, convolution2DNaiveHip, src_image, dst_image, kernel, width, height, kernel_width, kernel_height);
    
    testFunction("fftw convolution", repeat_times, convolution2DFFTW, src_image, dst_image, kernel, width, height, kernel_width, kernel_height);
    
    //for (int i = 0; i < 10; i++) {
    //    //printf("%f\n", dst_image[50 + i + width * 50]);
    //    printf("%f\n", dst_image[i]);
    //}
    //printf("\n");

    delete[] kernel;
    delete[] src_image;
    delete[] dst_image;
    return 0;
}
