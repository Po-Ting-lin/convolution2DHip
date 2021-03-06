#pragma once
#include <random>

inline static double gaussian(double x, double mu, double sigma) {
    return std::exp(-0.5 * ((x - mu) / sigma) * ((x - mu) / sigma));
}

static void getRandomImage(float* src, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        src[i] = (float)(rand() % 1000 / 100.0f);
    }
}

static void getGaussianKernel2D(float* kernel, int width, int height, double sigma) {
    const double w_mean = (int)(width / 2);
    const double h_mean = (int)(height / 2);
    float sum = 0.0f;
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            kernel[y * width + x] = gaussian(y, h_mean, sigma) * gaussian(x, w_mean, sigma);
            sum += kernel[y * width + x];
        }
    }
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            kernel[y * width + x] /= sum;
        }
    }
}
