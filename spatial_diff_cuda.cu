#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>

template<typename T>
__global__ void spatial_diff_kernel(T* out_data, const T* in_data,
                                   int ndims, const int* dims, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Calculate multi-dimensional indices from linear index
    int temp_idx = idx;
    const T center_in_data = in_data[idx];
    int coords[8]; // Support up to 8 dimensions

    for (int d = 0; d < ndims; d++) {
        coords[d] = temp_idx % dims[d];
        temp_idx /= dims[d];
    }
    // Copy coords to evaluate neighbor index
    int neighbor_coords[8];
    for (int d = 0; d < ndims; d++) {
        neighbor_coords[d] = coords[d];
    }
    // Compute spatial differences for each dimension
    for (int dim = 0; dim < ndims; dim++) {

        // Apply circular shift: move forward by 1 in current dimension
        neighbor_coords[dim] = (coords[dim] + 1) % dims[dim];

        // Convert back to linear index
        int neighbor_idx = 0;
        int stride = 1;
        for (int d = 0; d < ndims; d++) {
            neighbor_idx += neighbor_coords[d] * stride;
            stride *= dims[d];
        }

        // Compute difference: current - shifted
        out_data[dim + ndims * idx] = center_in_data - in_data[neighbor_idx];

        // roll back neighbor_coords
        neighbor_coords[dim] = coords[dim];
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
    const mwSize *in_dims = mxGPUGetDimensions(in_gpu);
    const mwSize *out_dims = mxGPUGetDimensions(out_gpu);
    int ndims_in = mxGPUGetNumberOfDimensions(in_gpu);
    int ndims_out = mxGPUGetNumberOfDimensions(out_gpu);

    // Validate dimensions
    if (ndims_out != ndims_in + 1) {
        mexErrMsgTxt("Output array must have one more dimension than input array");
    }
    if (out_dims[0] != ndims_in) {
        mexErrMsgTxt("First dimension of output must equal number of input dimensions");
    }
    for (int i = 0; i < ndims_in; i++) {
        if (out_dims[i + 1] != in_dims[i]) {
            mexErrMsgTxt("Dimension mismatch between input and output arrays");
        }
    }
    if (ndims_in > 8) {
        mexErrMsgTxt("The maximum dimension of input array is 8");
    }

    // Calculate total elements of input array
    int total_elements = 1;
    for (int i = 0; i < ndims_in; i++) {
        total_elements *= in_dims[i];
    }

    // Copy dimensions to device
    int *d_dims;
    int dims_host[8];
    for (int i = 0; i < ndims_in && i < 8; i++) {
        dims_host[i] = (int)in_dims[i];
    }

    cudaMalloc(&d_dims, ndims_in * sizeof(int));
    cudaMemcpy(d_dims, dims_host, ndims_in * sizeof(int), cudaMemcpyHostToDevice);

    // Create output array (copy of input)
    mxGPUArray *out_result = mxGPUCreateGPUArray(ndims_out, out_dims, in_class, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Launch kernel based on data type
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    if (in_class == mxDOUBLE_CLASS) {
        // Get pointers to data
        double *out_data = (double*)mxGPUGetData(out_result);
        const double *in_data = (const double*)mxGPUGetDataReadOnly(in_gpu);

        spatial_diff_kernel<double><<<blocksPerGrid, threadsPerBlock>>>(
            out_data, in_data, ndims_in, d_dims, total_elements);
    } else { // mxSINGLE_CLASS
        // Get pointers to data
        float *out_data = (float*)mxGPUGetData(out_result);
        const float *in_data = (const float*)mxGPUGetDataReadOnly(in_gpu);

        spatial_diff_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(
            out_data, in_data, ndims_in, d_dims, total_elements);
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