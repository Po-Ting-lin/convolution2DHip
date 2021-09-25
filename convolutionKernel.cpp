#include "convolutionKernel.h"

void convolution2DNaiveHip(float* src, float* dst, float* kernel, int iw, int ih, int kw, int kh) {
    if (kw % 2 == 0 || kh % 2 == 0) return;
    const int image_bytes_size = iw * ih * sizeof(float);
    const int kernel_bytes_size = kw * kh * sizeof(float);
    float* d_src;
    float* d_dst;
    float* d_kernel;
    hipMalloc((void**)&d_src, image_bytes_size);
    hipMalloc((void**)&d_dst, image_bytes_size);
    hipMalloc((void**)&d_kernel, kernel_bytes_size);
    hipMemcpy(d_src, src, image_bytes_size, hipMemcpyHostToDevice);
    hipMemcpy(d_kernel, kernel, kernel_bytes_size, hipMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(iDivUp(iw, BLOCK_DIM), iDivUp(ih, BLOCK_DIM));

    convolution2DNaiveHipKernel<<<grid, block>>>(d_src, d_dst, d_kernel, iw, ih, kw, kh);

    // this also will sync device
    hipMemcpy(dst, d_dst, image_bytes_size, hipMemcpyDeviceToHost);

    hipFree(d_src);
    hipFree(d_dst);
    hipFree(d_kernel);
}

__global__ void convolution2DNaiveHipKernel(float* dSrc, float* dDst, float* dKernel, int iw, int ih, int kw, int kh) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (x >= iw || y >= ih) return;
    const int reflect_num_x = kw / 2 / iw + 1;
    const int reflect_num_y = kh / 2 / ih + 1;
    const int k_center_y = kh / 2;
    const int k_center_x = kw / 2;

    dDst[y * iw + x] = 0.0f;
    for (int ky = 0; ky < kh; ky++) {
        for (int kx = 0; kx < kw; kx++) {
            int iy = y - k_center_y + ky;
            int ix = x - k_center_x + kx;

            // reflect the x-axis pixel 
            for (int k = 0; k < reflect_num_x; k++) {
                if (ix < 0) ix = -1 - ix;
                if (ix >= iw) ix = 2 * iw - ix - 1;
            }
            // reflect the y-axis pixel
            for (int k = 0; k < reflect_num_y; k++) {
                if (iy < 0) iy = -1 - iy;
                if (iy >= ih) iy = 2 * ih - iy - 1;
            }

            // convolve
            dDst[y * iw + x] += dSrc[iy * iw + ix] * dKernel[ky * kw + kx];
        }
    }
}
