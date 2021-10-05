#include "convolution.h"
#include "convolutionKernel.h"
#include "timer.h"
#include "utils.h"

static const char* checkCorrectness(float* testSrc, float* ref, int iWidth, int iHeight) {
    const float episilon = 1e-4;
    const char* result = "Pass";
    for (int i = 0; i < iWidth * iHeight; i++) {
        if (abs(ref[i] - testSrc[i]) > episilon) {
            result = "Fail";
	    break;
        }
    }
    return result;
}

template<typename F>
void testFunction(const char* name, int repeatTime, F func, float* src, float* dst, float* ref, float* kernel, int iWidth, int iHeight, int kWidth, int kHeight) {
    printf("%s: %.4f ms (average %d times)\n", name, funcTime(repeatTime, func, src, dst, kernel, iWidth, iHeight, kWidth, kHeight), repeatTime);
    printf("Correctness: %s\n\n", checkCorrectness(dst, ref, iWidth, iHeight));	
}

template<typename F>
void testFunction2(const char* name, int repeatTime, F func, float* src, float* dst, float* ref, float* row_kernel, float* col_kernel, int iWidth, int iHeight, int kWidth, int kHeight) {
    printf("%s: %.4f ms (average %d times)\n", name, funcTime(repeatTime, func, src, dst, row_kernel, col_kernel, iWidth, iHeight, kWidth, kHeight), repeatTime);
    printf("Correctness: %s\n\n", checkCorrectness(dst, ref, iWidth, iHeight));
}


int main() {
    gpuWarmUp();
    const int width = 1024;
    const int height = 1024;
    const int kernel_width = 31;
    const int kernel_height = 31;
    const double sigma = 5;

    float* kernel = new float[kernel_width * kernel_height];
    float* row_kernel = new float[kernel_width];
    float* col_kernel = new float[kernel_height];
    float* src_image = new float[width * height];
    float* dst_image = new float[width * height];
    float* ref_image = new float[width * height];
    getRandomImage(src_image, width, height);
    getGaussianKernel2D(kernel, kernel_width, kernel_height, sigma);
    getGaussianKernel2D(row_kernel, kernel_width, 1, sigma);
    getGaussianKernel2D(col_kernel, kernel_height, 1, sigma);
   
    // prepare ref_image for testing
    convolution2DOpencv(src_image, ref_image, kernel, width, height, kernel_width, kernel_height);
    
    const int repeat_times = 50;
    testFunction("Opencv conv", repeat_times, convolution2DOpencv, src_image, dst_image, ref_image, kernel, width, height, kernel_width, kernel_height);   
    testFunction("Naive conv", 1, convolution2DNaive, src_image, dst_image, ref_image, kernel, width, height, kernel_width, kernel_height);
    testFunction("Naive MP conv", 10, convolution2DNaiveMp, src_image, dst_image, ref_image, kernel, width, height, kernel_width, kernel_height);
    testFunction("FFTW conv", repeat_times, convolution2DFFTW, src_image, dst_image, ref_image, kernel, width, height, kernel_width, kernel_height);
    testFunction("Navie GPU conv", repeat_times, convolution2DNaiveHip, src_image, dst_image, ref_image, kernel, width, height, kernel_width, kernel_height);
    testFunction2("Seperable GPU conv", repeat_times, convolution2DSepHip, src_image, dst_image, ref_image, row_kernel, col_kernel, width, height, kernel_width, kernel_height);
    testFunction2("Seperable Const GPU conv", repeat_times, convolution2DSepConstHip, src_image, dst_image, ref_image, row_kernel, col_kernel, width, height, kernel_width, kernel_height);

    assertGPUParameters(width, height, kernel_width, kernel_height);

    testFunction2("Seperable Const Smem GPU conv", repeat_times, convolution2DSepConstSmemHip, src_image, dst_image, ref_image, row_kernel, col_kernel, width, height, kernel_width, kernel_height);
    testFunction2("Seperable Const Smem Unroll GPU conv", repeat_times, convolution2DSepConstSmemUnrollHip, src_image, dst_image, ref_image, row_kernel, col_kernel, width, height, kernel_width, kernel_height);
    
    //for (int i = 0; i < 10; i++) {
        //printf("pred: %f; ref: %f\n", dst_image[500 + i + width * 220], ref_image[500 + i + width * 220]);
        //printf("pred: %f; ref: %f\n", dst_image[i], ref_image[i]);
    //}
    //printf("\n");
    
    gpuReset();
    delete[] kernel;
    delete[] row_kernel;
    delete[] col_kernel;
    delete[] src_image;
    delete[] dst_image;
    delete[] ref_image;
    return 0;
}
