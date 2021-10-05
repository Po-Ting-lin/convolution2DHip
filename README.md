# Convolution 2D

### Introduction
Implementation of convolution in different ways. GPU code implemented in Heterogeneous-Computing Interface for Portability (HIP), which is a C++ Runtime API and Kernel Language that allows us to apply on AMD and NVIDIA GPUs from single source code. 


### Test   
* Random matrix: 1024 * 1024
* Gaussain kernel: 31 * 31
* Border: BORDER_REFLECT (fedcba|abcdefgh|hgfedcb)
* All of the methods validated by opencv filter2D.   
```
Opencv conv: 10.8281 ms (average 50 times)

Naive conv: 6764.9290 ms (average 1 times)
Correctness: Pass

Naive OpenMP conv: 1151.7960 ms (average 10 times)
Correctness: Pass

FFTW conv: 28.9246 ms (average 50 times)
Correctness: Pass

Navie GPU conv: 5.9703 ms (average 50 times)
Correctness: Pass

Seperable GPU conv: 1.6737 ms (average 50 times)
Correctness: Pass

Seperable Const GPU conv: 1.5371 ms (average 50 times)
Correctness: Pass

Seperable Const Smem GPU conv: 1.2895 ms (average 50 times)
Correctness: Pass
```

### Reference
