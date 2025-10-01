clear; clc;
mexcuda('-largeArrayDims', 'lp_total_variation_cuda.cu');
mexcuda('-output', 'helper_fista_TV_inner_gpu', 'helper_fista_TV_inner_cuda.cu', 'helper_fista_TV_inner_mex.cpp');