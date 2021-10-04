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

    convolution2DSepConstColHipKernel<<<grid, block>>>(d_src, d_mid_dst, iw, ih, kh);
    hipDeviceSynchronize();
    convolution2DSepConstRowHipKernel<<<grid, block>>>(d_mid_dst, d_dst, iw, ih, kw);
    hipMemcpy(dst, d_dst, image_bytes_size, hipMemcpyDeviceToHost);

    hipFree(d_src);
    hipFree(d_dst);
    hipFree(d_mid_dst);
}

#define RowBlockDim_x 16
#define RowBlockDim_y 4
#define RowStep 8
#define RowPadStep 1
#define ColBlockDim_x 16
#define ColBlockDim_y 8
#define ColStep 8
#define ColPadStep 1

void assertGPUParameters(int iw, int ih, int kw, int kh){
    assert(ColBlockDim_y * ColPadStep >= kh / 2);
    assert(ih % (ColStep * ColBlockDim_y) == 0);
    assert(iw % ColBlockDim_x == 0);
    assert(RowBlockDim_x * RowPadStep >= kw / 2);
    assert(ih % RowBlockDim_y == 0);
    assert(iw % (RowStep * RowBlockDim_x) == 0);
}

void convolution2DSepConstSmemHip(float* src, float* dst, float* rowKernel, float* colKernel, int iw, int ih, int kw, int kh){
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

    dim3 row_block(RowBlockDim_x, RowBlockDim_y);
    dim3 col_block(ColBlockDim_x, ColBlockDim_y);
    dim3 row_grid(iDivUp(iw, RowBlockDim_x * RowStep), iDivUp(ih, RowBlockDim_y));
    dim3 col_grid(iDivUp(iw, ColBlockDim_x), iDivUp(ih, ColBlockDim_y * ColStep));

    convolution2DSepConstSmemColHipKernel<<<col_grid, col_block>>>(d_src, d_mid_dst, iw, ih, kh);
    hipDeviceSynchronize();
    convolution2DSepConstSmemRowHipKernel<<<row_grid, row_block>>>(d_mid_dst, d_dst, iw, ih, kw);
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

__global__ void convolution2DSepConstColHipKernel(float* dSrc, float* dDst, int iw, int ih, int kh){
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
	dDst[y * iw + x] += dSrc[iy * iw + ix] * c_col_kernel[ky];
    }  
}

__global__ void convolution2DSepConstRowHipKernel(float* dSrc, float* dDst, int iw, int ih, int kw){
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
        dDst[y * iw + x] += dSrc[iy * iw + ix] * c_row_kernel[kx];
    }
}

__global__ void convolution2DSepConstSmemColHipKernel(float* dSrc, float* dDst, int iw, int ih, int kh){
    __shared__ float smem[ColBlockDim_x][(ColStep + 2 * ColPadStep) * ColBlockDim_y + 1];
    const int kernelHalf = kh / 2;
    const int reflect_num_y = kh / 2 / ih + 1;
    const int x = hipBlockIdx_x * ColBlockDim_x + hipThreadIdx_x;
    const int y = hipBlockIdx_y * ColStep * ColBlockDim_y + hipThreadIdx_y - ColPadStep * ColBlockDim_y;
    
    // block thread map into image block;
    float* mdSrc = dSrc + y * iw + x;
    float* mdDst = dDst + y * iw + x;

    // load main src into smem
    for (int i = ColPadStep; i < ColPadStep + ColStep; i++){
        smem[hipThreadIdx_x][i * ColBlockDim_y + hipThreadIdx_y] = mdSrc[i * ColBlockDim_y * iw];
    }
    // load src into top pad smem
    for (int i = 0; i < ColPadStep; i++){
	int ry = y;
	for (int k = 0; k < reflect_num_y; k++) {
            if (ry < 0) ry = -1 - ry;
            if (ry >= ih) ry = 2 * ih - ry - 1;
        }
        smem[hipThreadIdx_x][i * ColBlockDim_y + hipThreadIdx_y] = dSrc[ry * iw + x + i * ColBlockDim_y * iw];
    }
    // load src into bottom pad smem
    for (int i = ColStep; i < ColStep + ColPadStep; i++){
        int ry = y;
	for (int k = 0; k < reflect_num_y; k++) {
            if (ry < 0) ry = -1 - ry;
            if (ry >= ih) ry = 2 * ih - ry - 1;
        }
        smem[hipThreadIdx_x][i * ColBlockDim_y + hipThreadIdx_y] = dSrc[ry * iw + x + i * ColBlockDim_y * iw];
    }

    __syncthreads();

    for (int i = ColPadStep; i < ColPadStep + ColStep; i++){
        float sum = 0.0f;
	for (int k = -kernelHalf; k <= kernelHalf; k++){
	    sum += c_col_kernel[k + kernelHalf] * smem[hipThreadIdx_x][i * ColBlockDim_y + hipThreadIdx_y + k];
	}
	mdDst[i * ColBlockDim_y * iw] = sum;
    }
}

__global__ void convolution2DSepConstSmemRowHipKernel(float* dSrc, float* dDst, int iw, int ih, int kw){
    // iw, ih % 16 == 0!!
    __shared__ float smem[RowBlockDim_y][(RowPadStep * 2 + RowStep) * RowBlockDim_x];
    const int kernelHalf = kw / 2;
    const int reflect_num_x = kw / 2 / iw + 1;
    const int x = hipBlockIdx_x * RowStep * RowBlockDim_x + hipThreadIdx_x - RowPadStep * RowBlockDim_x;
    const int y = hipBlockIdx_y * RowBlockDim_y + hipThreadIdx_y;
    
    // block thread map into image block
    float* mdSrc = dSrc + y * iw + x;
    float* mdDst = dDst + y * iw + x;

    // load main src into smem
    for (int i = RowPadStep; i < RowPadStep + RowStep; i++){
        smem[hipThreadIdx_y][i * RowBlockDim_x + hipThreadIdx_x] = mdSrc[i * RowBlockDim_x];
    }
    // load src into left pad smem
    for (int i = 0; i < RowPadStep; i++){
	int rx = x;
        for (int k = 0; k < reflect_num_x; k++) {
            if (rx < 0) rx = -1 - rx;
            if (rx >= iw) rx = 2 * iw - rx - 1;
        }
        smem[hipThreadIdx_y][i * RowBlockDim_x + hipThreadIdx_x] = dSrc[y * iw + rx + i * RowBlockDim_x];
    }
    // load src into right pad smem
    for (int i = RowStep; i < RowStep + RowPadStep; i++){
        int rx = x;
        for (int k = 0; k < reflect_num_x; k++) {
            if (rx < 0) rx = -1 - rx;
            if (rx >= iw) rx = 2 * iw - rx - 1;
        }
	smem[hipThreadIdx_y][i * RowBlockDim_x + hipThreadIdx_x] = dSrc[y * iw + rx + i * RowBlockDim_x];
    }

    __syncthreads();

    for (int i = RowPadStep; i < RowPadStep + RowStep; i++){
        float sum = 0.0f;
	for (int k = -kernelHalf; k <= kernelHalf; k++){
	    sum += smem[hipThreadIdx_y][i * RowBlockDim_x + hipThreadIdx_x + k] * c_row_kernel[k + kernelHalf];
	}
	mdDst[i * hipBlockDim_x] = sum;
    }
}
