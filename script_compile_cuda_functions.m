clear; clc;
mexcuda('-largeArrayDims', 'spatial_diff_cuda.cu');
mexcuda('-largeArrayDims', 'spatial_diff_T_cuda.cu');
mexcuda('-largeArrayDims', 'lp_total_variation_cuda.cu');