#include "convolution.h"

void convolution2DOpencv(float* src, float* dst, float* kernel, int iWidth, int iHeight, int kWidth, int kHeight) {
    cv::Mat src_mat(iHeight, iWidth, CV_32FC1, src);
    cv::Mat dst_mat(iHeight, iWidth, CV_32FC1, dst);
    cv::Mat kernel_mat(kHeight, kWidth, CV_32FC1, kernel);
    cv::filter2D(src_mat, dst_mat, CV_32FC1, kernel_mat, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
}

void convolution2DNaive(float* src, float* dst, float* kernel, int iWidth, int iHeight, int kWidth, int kHeight)
{
    if (kWidth % 2 == 0 || kHeight % 2 == 0) return;
    const int reflect_num_x = kWidth / 2 / iWidth + 1;
    const int reflect_num_y = kHeight / 2 / iHeight + 1;
    const int k_center_y = kHeight / 2;
    const int k_center_x = kWidth / 2;

    for (int dy = 0; dy < iHeight; dy++) {
        for (int dx = 0; dx < iWidth; dx++) {
            dst[dy * iWidth + dx] = 0.0f;
            for (int ky = 0; ky < kHeight; ky++) {
                for (int kx = 0; kx < kWidth; kx++) {
                    int iy = dy - k_center_y + ky;
                    int ix = dx - k_center_x + kx;

                    // reflect the x-axis pixel 
                    for (int k = 0; k < reflect_num_x; k++) {
                        if (ix < 0) ix = -1 - ix;
                        if (ix >= iWidth) ix = 2 * iWidth - ix - 1;
                    }
                    // reflect the y-axis pixel
                    for (int k = 0; k < reflect_num_y; k++) {
                        if (iy < 0) iy = -1 - iy;
                        if (iy >= iHeight) iy = 2 * iHeight - iy - 1;
                    }

                    // convolve
                    dst[dy * iWidth + dx] += src[iy * iWidth + ix] * kernel[ky * kWidth + kx];
                }
            }
        }
    }
}

void convolution2DNaiveMp(float* src, float* dst, float* kernel, int iWidth, int iHeight, int kWidth, int kHeight)
{
    if (kWidth % 2 == 0 || kHeight % 2 == 0) return;
    const int reflect_num_x = kWidth / 2 / iWidth + 1;
    const int reflect_num_y = kHeight / 2 / iHeight + 1;
    const int k_center_y = kHeight / 2;
    const int k_center_x = kWidth / 2;

#pragma omp parallel for collapse(2)
    for (int dy = 0; dy < iHeight; dy++) {
        for (int dx = 0; dx < iWidth; dx++) {
            dst[dy * iWidth + dx] = 0.0f;
            for (int ky = 0; ky < kHeight; ky++) {
                for (int kx = 0; kx < kWidth; kx++) {
                    int iy = dy - k_center_y + ky;
                    int ix = dx - k_center_x + kx;

                    // reflect the x-axis pixel 
                    for (int k = 0; k < reflect_num_x; k++) {
                        if (ix < 0) ix = -1 - ix;
                        if (ix >= iWidth) ix = 2 * iWidth - ix - 1;
                    }
                    // reflect the y-axis pixel
                    for (int k = 0; k < reflect_num_y; k++) {
                        if (iy < 0) iy = -1 - iy;
                        if (iy >= iHeight) iy = 2 * iHeight - iy - 1;
                    }

                    // convolve
                    dst[dy * iWidth + dx] += src[iy * iWidth + ix] * kernel[ky * kWidth + kx];
                }
            }
        }
    }
}

void convolution2DFFTW(float* src, float* dst, float* kernel, int iWidth, int iHeight, int kWidth, int kHeight) {
    int FFTW_FACTORS[7] = { 13,11,7,5,3,2,0 };
    const int padded_height = iHeight + kHeight - 1;
    const int padded_width = iWidth + kWidth - 1;
    const int halfk_x = kWidth >> 1;
    const int halfk_y = kHeight >> 1;
    const int reflect_num_x = halfk_x / iWidth + 1;
    const int reflect_num_y = halfk_y / iHeight + 1;
    const int crop_x_offset = halfk_x << 1;
    const int crop_y_offset = halfk_y << 1;
    const int fftw_height = find_closest_factor(padded_height + kHeight / 2, FFTW_FACTORS);
    const int fftw_width = find_closest_factor(padded_width + kWidth / 2, FFTW_FACTORS);
    const int t_domain_size = fftw_height * fftw_width;
    const int f_domain_size = fftw_height * (fftw_width / 2 + 1);
    if (fftw_height <= 0 || fftw_width <= 0) return;
    float* in_src = new float[t_domain_size];
    float* out_src = (float*)fftwf_malloc(sizeof(fftwf_complex) * f_domain_size);
    float* in_kernel = new float[t_domain_size];
    float* out_kernel = (float*)fftwf_malloc(sizeof(fftwf_complex) * f_domain_size);
    float* dst_fft = new float[t_domain_size];
    float* product = (float*)fftwf_malloc(sizeof(fftwf_complex) * f_domain_size);

    int result = fftwf_import_wisdom_from_filename("w.wisdom");
    fftwf_plan p_forw_src = fftwf_plan_dft_r2c_2d(fftw_height, fftw_width, in_src, (fftwf_complex*)out_src, FFTW_MEASURE);
    fftwf_plan p_forw_kernel = fftwf_plan_dft_r2c_2d(fftw_height, fftw_width, in_kernel, (fftwf_complex*)out_kernel, FFTW_MEASURE);
    fftwf_plan p_back = fftwf_plan_dft_c2r_2d(fftw_height, fftw_width, (fftwf_complex*)product, dst_fft, FFTW_MEASURE);
    
    //fftwf_export_wisdom_to_filename("w.wisdom");
    
    float* ptr, * ptr_end, * ptr2;
    memset(in_src, 0, t_domain_size * sizeof(float));
    memset(in_kernel, 0, t_domain_size * sizeof(float));
    // reflect pad image
    for (int y = 0; y < padded_height; y++) {
        for (int x = 0; x < padded_width; x++) {
            int src_x = x - halfk_x;
            int src_y = y - halfk_y;
            // reflect k times 
            for (int k = 0; k < reflect_num_x; k++) {
                if (src_x < 0) src_x = -1 - src_x;
                else if (src_x >= padded_width - 2 * halfk_x) src_x = 2 * iWidth - src_x - 1;
            }
            for (int k = 0; k < reflect_num_y; k++) {
                if (src_y < 0) src_y = -1 - src_y;
                else if (src_y >= padded_height - 2 * halfk_y) src_y = 2 * iHeight - src_y - 1;
            }
            in_src[y * fftw_width + x] = src[src_y * iWidth + src_x];
        }
    }
    for (int y = 0; y < kHeight; y++)
        for (int x = 0; x < kWidth; x++)
            in_kernel[y * fftw_width + x] = kernel[y * kWidth + x];
    fftwf_execute(p_forw_src);
    fftwf_execute(p_forw_kernel);
    for (int i = 0; i < 2 * f_domain_size; i += 2) {
        product[i] = out_src[i] * out_kernel[i] - out_src[i + 1] * out_kernel[i + 1];
        product[i + 1] = out_src[i] * out_kernel[i + 1] + out_src[i + 1] * out_kernel[i];
    }
    fftwf_execute(p_back);
    for (ptr = dst_fft, ptr_end = dst_fft + t_domain_size; ptr != ptr_end; ++ptr)
        *ptr /= (float)t_domain_size;
    for (int y = 0; y < iHeight; y++) {
        for (int x = 0; x < iWidth; x++) {
            dst[y * iWidth + x] = dst_fft[(crop_y_offset + y) * fftw_width + (crop_x_offset + x)];
        }
    }
    delete[] in_src;
    delete[] in_kernel;
    delete[] dst_fft;
    fftwf_free((fftwf_complex*)out_src);
    fftwf_free((fftwf_complex*)out_kernel);
    fftwf_free((fftwf_complex*)product);
    fftwf_destroy_plan(p_forw_src);
    fftwf_destroy_plan(p_forw_kernel);
    fftwf_destroy_plan(p_back);
}

