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

void convolution2DSepHip(float* src, float* dst, float* rowKernel, float* colKernel, int iw, int ih, int kw, int kh){
    if (kw % 2 == 0 || kh % 2 == 0) return;
    const int image_bytes_size = iw * ih * sizeof(float);
    const int row_kernel_bytes_size = kw * sizeof(float);
    const int col_kernel_bytes_size = kh * sizeof(float);
    float* d_src;
    float* d_dst;
    float* d_mid_dst;
    float* d_row_kernel;
    float* d_col_kernel;
    hipMalloc((void**)&d_src, image_bytes_size);
    hipMalloc((void**)&d_dst, image_bytes_size);
    hipMalloc((void**)&d_mid_dst, image_bytes_size);
    hipMalloc((void**)&d_row_kernel, row_kernel_bytes_size);
    hipMalloc((void**)&d_col_kernel, col_kernel_bytes_size);
    hipMemcpy(d_src, src, image_bytes_size, hipMemcpyHostToDevice);
    hipMemcpy(d_row_kernel, rowKernel, row_kernel_bytes_size, hipMemcpyHostToDevice);
    hipMemcpy(d_col_kernel, colKernel, col_kernel_bytes_size, hipMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(iDivUp(iw, BLOCK_DIM), iDivUp(ih, BLOCK_DIM));

    convolution2DSepColHipKernel<<<grid, block>>>(d_src, d_mid_dst, d_col_kernel, iw, ih, kh);
    hipDeviceSynchronize();
    convolution2DSepRowHipKernel<<<grid, block>>>(d_mid_dst, d_dst, d_row_kernel, iw, ih, kw);
    hipMemcpy(dst, d_dst, image_bytes_size, hipMemcpyDeviceToHost);

    hipFree(d_src);
    hipFree(d_dst);
    hipFree(d_mid_dst);
    hipFree(d_row_kernel);
    hipFree(d_col_kernel);
}

__constant__ float c_row_kernel[31];
__constant__ float c_col_kernel[31];

void convolution2DSepConstHip(float* src, float* dst, float* rowKernel, float* colKernel, int iw, int ih, int kw, int kh){
    if (kw % 2 == 0 || kh % 2 == 0) return;
    const int image_bytes_size = iw * ih * sizeof(float);
    const int row_kernel_bytes_size = kw * sizeof(float);
    const int col_kernel_bytes_size = kh * sizeof(float);
    float* d_src;
    float* d_dst;
    float* d_mid_dst;
    hipMalloc((void**)&d_src, image_bytes_size);
    hipMalloc((void**)&d_dst, image_bytes_size);
    hipMalloc((void**)&d_mid_dst, image_bytes_size);
    hipMemcpy(d_src, src, image_bytes_size, hipMemcpyHostToDevice);
    hipMemcpyToSymbol(c_row_kernel, rowKernel, row_kernel_bytes_size);
    hipMemcpyToSymbol(c_col_kernel, colKernel, col_kernel_bytes_size);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(iDivUp(iw, BLOCK_DIM), iDivUp(ih, BLOCK_DIM));

    convolution2DSepColHipKernel<<<grid, block>>>(d_src, d_mid_dst, &c_col_kernel[0], iw, ih, kh);
    hipDeviceSynchronize();
    convolution2DSepRowHipKernel<<<grid, block>>>(d_mid_dst, d_dst, &c_row_kernel[0], iw, ih, kw);
    hipMemcpy(dst, d_dst, image_bytes_size, hipMemcpyDeviceToHost);

    hipFree(d_src);
    hipFree(d_dst);
    hipFree(d_mid_dst);
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

__global__ void convolution2DSepColHipKernel(float* dSrc, float* dDst, float* dKernel, int iw, int ih, int kh){
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (x >= iw || y >= ih) return;
    const int reflect_num_y = kh / 2 / ih + 1;
    const int k_center_y = kh / 2;

    dDst[y * iw + x] = 0.0f;
    for (int ky = 0; ky < kh; ky++) {
        int iy = y - k_center_y + ky;
        int ix = x;
        // reflect the y-axis pixel
	for (int k = 0; k < reflect_num_y; k++) {
	    if (iy < 0) iy = -1 - iy;
	    if (iy >= ih) iy = 2 * ih - iy - 1;
	}
	// convolve
	dDst[y * iw + x] += dSrc[iy * iw + ix] * dKernel[ky];
    }  
}

__global__ void convolution2DSepRowHipKernel(float* dSrc, float* dDst, float* dKernel, int iw, int ih, int kw){
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (x >= iw || y >= ih) return;
    const int reflect_num_x = kw / 2 / iw + 1;
    const int k_center_x = kw / 2;

    dDst[y * iw + x] = 0.0f;
    for (int kx = 0; kx < kw; kx++) {
        int iy = y;
        int ix = x - k_center_x + kx;
        // reflect the x-axis pixel 
        for (int k = 0; k < reflect_num_x; k++) {
            if (ix < 0) ix = -1 - ix;
            if (ix >= iw) ix = 2 * iw - ix - 1;
        }
        // convolve
        dDst[y * iw + x] += dSrc[iy * iw + ix] * dKernel[kx];
    }
}


