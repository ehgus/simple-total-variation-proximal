#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>

template<typename T>
__global__ void spatial_diff_T_kernel(T* out_data, const T* in_data,
                                      int ndims, const int* dims, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Pre-compute strides for efficient indexing
    int strides[8];
    strides[0] = 1;
    for (int d = 1; d < ndims; d++) {
        strides[d] = strides[d-1] * dims[d-1];
    }

    // Calculate multi-dimensional coordinates from linear index
    int temp_idx = idx;
    int coords[8];
    for (int d = 0; d < ndims; d++) {
        coords[d] = temp_idx % dims[d];
        temp_idx /= dims[d];
    }

    // Initialize output with sum across first dimension (sum of in_arr over dim 0)
    T sum_val = 0.0;
    for (int dim = 0; dim < ndims; dim++) {
        sum_val += in_data[dim + ndims * idx];
    }
    out_data[idx] = sum_val;

    // Compute transpose spatial differences for each dimension
    for (int dim = 0; dim < ndims; dim++) {
        // Apply backward circular shift: move backward by 1 in current dimension
        int neighbor_coord = (coords[dim] - 1 + dims[dim]) % dims[dim];

        // Calculate neighbor index efficiently using pre-computed strides
        int neighbor_idx = idx + (neighbor_coord - coords[dim]) * strides[dim];

        // Subtract the shifted value (transpose operation)
        out_data[idx] -= in_data[dim + ndims * neighbor_idx];
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Initialize GPU
    mxInitGPU();

    // Input validation
    if (nrhs != 2) {
        mexErrMsgTxt("Two inputs required: out_arr and in_arr");
    }
    if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments");
    }

    // Get input arrays
    const mxGPUArray *out_gpu = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray *in_gpu = mxGPUCreateFromMxArray(prhs[1]);

    // Validate data types
    mxClassID out_class = mxGPUGetClassID(out_gpu);
    mxClassID in_class = mxGPUGetClassID(in_gpu);

    if (out_class != in_class) {
        mexErrMsgTxt("Input and output arrays must have the same data type");
    }

    if (in_class != mxDOUBLE_CLASS && in_class != mxSINGLE_CLASS) {
        mexErrMsgTxt("Input arrays must be of type double or single");
    }

    // Get dimensions
    const mwSize *out_dims = mxGPUGetDimensions(out_gpu);
    const mwSize *in_dims = mxGPUGetDimensions(in_gpu);
    int ndims_out = mxGPUGetNumberOfDimensions(out_gpu);
    int ndims_in = mxGPUGetNumberOfDimensions(in_gpu);

    // Validate dimensions
    if (ndims_in != ndims_out + 1) {
        mexErrMsgTxt("Input array must have one more dimension than output array");
    }
    if (in_dims[0] != ndims_out) {
        mexErrMsgTxt("First dimension of input must equal number of output dimensions");
    }
    for (int i = 0; i < ndims_out; i++) {
        if (in_dims[i + 1] != out_dims[i]) {
            mexErrMsgTxt("Dimension mismatch between input and output arrays");
        }
    }
    if (ndims_out > 8) {
        mexErrMsgTxt("The maximum dimension of output array is 8");
    }

    // Calculate total elements of output array
    int total_elements = 1;
    for (int i = 0; i < ndims_out; i++) {
        total_elements *= out_dims[i];
    }

    // Copy dimensions to device
    int *d_dims;
    int dims_host[8];
    for (int i = 0; i < ndims_out && i < 8; i++) {
        dims_host[i] = (int)out_dims[i];
    }

    cudaMalloc(&d_dims, ndims_out * sizeof(int));
    cudaMemcpy(d_dims, dims_host, ndims_out * sizeof(int), cudaMemcpyHostToDevice);

    // Create output array
    mxGPUArray *out_result = mxGPUCreateGPUArray(ndims_out, out_dims, in_class, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Launch kernel based on data type
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    if (in_class == mxDOUBLE_CLASS) {
        // Get pointers to data
        double *out_data = (double*)mxGPUGetData(out_result);
        const double *in_data = (const double*)mxGPUGetDataReadOnly(in_gpu);

        spatial_diff_T_kernel<double><<<blocksPerGrid, threadsPerBlock>>>(
            out_data, in_data, ndims_out, d_dims, total_elements);
    } else { // mxSINGLE_CLASS
        // Get pointers to data
        float *out_data = (float*)mxGPUGetData(out_result);
        const float *in_data = (const float*)mxGPUGetDataReadOnly(in_gpu);

        spatial_diff_T_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(
            out_data, in_data, ndims_out, d_dims, total_elements);
    }

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexErrMsgTxt("CUDA kernel execution failed");
    }

    // Wait for completion
    cudaDeviceSynchronize();

    // Clean up device memory
    cudaFree(d_dims);

    // Create output
    plhs[0] = mxGPUCreateMxArrayOnGPU(out_result);

    // Clean up
    mxGPUDestroyGPUArray(out_gpu);
    mxGPUDestroyGPUArray(in_gpu);
    mxGPUDestroyGPUArray(out_result);
}