#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cmath>

// Template helper functions for CUDA math operations
template<typename T>
__device__ T cuda_pow(T base, T exponent);

template<>
__device__ float cuda_pow<float>(float base, float exponent) {
    return powf(base, exponent);
}

template<>
__device__ double cuda_pow<double>(double base, double exponent) {
    return pow(base, exponent);
}

// Device function: spatial_diff_T operation
// Input: z_data (input array ndims, total_elements), dims, strides, ndims
// Output: returns the result
template<typename T>
__device__ T spatial_diff_T_device(const T* z_data, int idx, int ndims, const int* dims, const int* strides) {
    // Calculate multi-dimensional coordinates from linear index
    int temp_idx = idx;
    int coords[8];
    for (int d = 0; d < ndims; d++) {
        coords[d] = temp_idx % dims[d];
        temp_idx /= dims[d];
    }

    // Initialize with sum across first dimension (sum of z over dim 0)
    T sum_val = T(0.0);
    for (int dim = 0; dim < ndims; dim++) {
        sum_val += z_data[dim + ndims * idx];
    }

    // Compute transpose spatial differences for each dimension
    for (int dim = 0; dim < ndims; dim++) {
        // Apply backward circular shift: move backward by 1 in current dimension
        int neighbor_coord = (coords[dim] - 1 + dims[dim]) % dims[dim];

        // Calculate neighbor index efficiently using pre-computed strides
        int neighbor_idx = idx + (neighbor_coord - coords[dim]) * strides[dim];

        // Subtract the shifted value (transpose operation)
        sum_val -= z_data[dim + ndims * neighbor_idx];
    }

    // Return result
    return sum_val;
}

// Device function: spatial_diff operation
// Input: x_tmp (total_elements), dims, strides, ndims
// Output: ndims values for position idx (stored in result array)
template<typename T>
__device__ void spatial_diff_device(const T* x_tmp_data, T* result, int idx, int ndims, const int* dims, const int* strides) {
    const T center_data = x_tmp_data[idx];

    // Calculate multi-dimensional coordinates from linear index
    int temp_idx = idx;
    int coords[8];
    for (int d = 0; d < ndims; d++) {
        coords[d] = temp_idx % dims[d];
        temp_idx /= dims[d];
    }

    // Compute spatial differences for each dimension
    for (int dim = 0; dim < ndims; dim++) {
        // Apply circular shift: move forward by 1 in current dimension
        int neighbor_coord = (coords[dim] + 1) % dims[dim];

        // Calculate neighbor index efficiently using pre-computed strides
        int neighbor_idx = idx + (neighbor_coord - coords[dim]) * strides[dim];

        // Compute difference: current - shifted
        result[dim] = center_data - x_tmp_data[neighbor_idx];
    }
}

// Device function: L2 norm projection
// Input: z_element, z_tmp_element (ndims values each)
// Output: projected z_element (ndims values)
template<typename T>
__device__ void l2_norm_projection_device(T* z_element, const T* z_tmp_element, int ndims) {
    // Calculate the L2 norm of z_tmp_element
    T norm_val = T(0.0);
    for (int d = 0; d < ndims; d++) {
        norm_val += z_tmp_element[d] * z_tmp_element[d];
    }
    norm_val = cuda_pow(norm_val, T(0.5)); // sqrt

    // Apply L2 projection: z = z_tmp / max(1, ||z_tmp||_2)
    T scale_factor = (norm_val > T(1.0)) ? norm_val : T(1.0);
    for (int d = 0; d < ndims; d++) {
        z_element[d] = z_tmp_element[d] / scale_factor;
    }
}

// Device function: L-infinity norm projection
// Input: z_element, z_tmp_element (ndims values each)
// Output: projected z_element (ndims values)
template<typename T>
__device__ void linf_norm_projection_device(T* z_element, const T* z_tmp_element, int ndims) {
    // L-infinity projection: clamp each component to [-1, 1]
    // y = max(min(x, 1), -1)
    for (int d = 0; d < ndims; d++) {
        T val = z_tmp_element[d];
        if (val > T(1.0)) {
            z_element[d] = T(1.0);
        } else if (val < T(-1.0)) {
            z_element[d] = T(-1.0);
        } else {
            z_element[d] = val;
        }
    }
}

// Device function: Generic projection dispatcher
// Input: z_element, z_tmp_element (ndims values each), projection_type
// Output: projected z_element (ndims values)
template<typename T>
__device__ void unit_ball_projection_device(T* z_element, const T* z_tmp_element, int ndims, int projection_type) {
    if (projection_type == 2) {
        // L2 projection
        l2_norm_projection_device(z_element, z_tmp_element, ndims);
    } else if (projection_type == -1) {
        // L-infinity projection (use -1 to represent infinity)
        linf_norm_projection_device(z_element, z_tmp_element, ndims);
    } else {
        // Default to L2 projection
        l2_norm_projection_device(z_element, z_tmp_element, ndims);
    }
}

// Kernel 1: spatial_diff_T operation
template<typename T>
__global__ void spatial_diff_T_kernel(
    T* x_tmp_data,          // Output: x_tmp array (total_elements)
    const T* x_data,        // Input: x (total_elements)
    const T* z_data,        // Input: z (ndims, total_elements)
    T w,                    // Weight parameter
    int ndims,              // Number of dimensions
    const int* dims,        // Dimension sizes
    const int* strides,     // Pre-computed strides
    int total_elements      // Total number of elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Step 1: x_tmp = x + w * spatial_diff_T(z)
    T spatial_diff_T_result = spatial_diff_T_device(z_data, idx, ndims, dims, strides);
    x_tmp_data[idx] = x_data[idx] + w * spatial_diff_T_result;
}

// Kernel 2: spatial_diff operation and z update
template<typename T>
__global__ void spatial_diff_z_update_kernel(
    T* z_data,              // Input/Output: z (ndims, total_elements)
    const T* x_tmp_data,    // Input: x_tmp array (total_elements)
    T v,                    // Norm weight parameter
    int projection_type,    // Projection type: 2 for L2, -1 for L-infinity
    int ndims,              // Number of dimensions
    const int* dims,        // Dimension sizes
    const int* strides,     // Pre-computed strides
    int total_elements      // Total number of elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Local arrays for this thread's computation
    T z_tmp_local[8];  // Support up to 8 dimensions
    T spatial_diff_result[8];

    // Step 2: z_tmp = z - v * spatial_diff(x_tmp)
    spatial_diff_device(x_tmp_data, spatial_diff_result, idx, ndims, dims, strides);
    for (int d = 0; d < ndims; d++) {
        z_tmp_local[d] = z_data[d + ndims * idx] - v * spatial_diff_result[d];
    }

    // Step 3: z = v * norm.projection(z, z_tmp / v)
    T z_tmp_scaled[8];
    for (int d = 0; d < ndims; d++) {
        z_tmp_scaled[d] = z_tmp_local[d] / v;
    }

    T z_projected[8];
    unit_ball_projection_device(z_projected, z_tmp_scaled, ndims, projection_type);

    for (int d = 0; d < ndims; d++) {
        z_data[d + ndims * idx] = v * z_projected[d];
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Initialize GPU
    mxInitGPU();

    // Input validation
    if (nrhs != 5) {
        mexErrMsgTxt("Five inputs required: x, w, v, niter, p_norm");
    }
    if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments");
    }

    // Get input arrays and parameters
    const mxGPUArray *x_gpu = mxGPUCreateFromMxArray(prhs[0]);
    double w = mxGetScalar(prhs[1]);
    double v = mxGetScalar(prhs[2]);
    int niter = (int)mxGetScalar(prhs[3]);
    double p_norm = mxGetScalar(prhs[4]);

    // Determine projection type from p_norm
    // Following MATLAB logic: obj.norm = LpUnitBall(round(1/(1-1/p)))
    int projection_type;
    if (isinf(p_norm)) {
        projection_type = -1;  // L-infinity projection
    } else {
        double dual_p = 1.0 / (1.0 - 1.0 / p_norm);
        int rounded_dual_p = (int)round(dual_p);
        if (rounded_dual_p == 2) {
            projection_type = 2;   // L2 projection
        } else if (isinf(dual_p)) {
            projection_type = -1;  // L-infinity projection
        } else {
            mexErrMsgTxt("p value other than 0 and Inf is not supported");
            projection_type = 2;   // Default to L2 projection
        }
    }

    // Validate data types
    mxClassID x_class = mxGPUGetClassID(x_gpu);
    if (x_class != mxDOUBLE_CLASS && x_class != mxSINGLE_CLASS) {
        mexErrMsgTxt("Input array must be of type double or single");
    }

    // Get dimensions of input x
    const mwSize *x_dims = mxGPUGetDimensions(x_gpu);
    int ndims_x = mxGPUGetNumberOfDimensions(x_gpu);

    if (ndims_x > 8) {
        mexErrMsgTxt("Maximum 8 dimensions supported");
    }

    // Calculate total elements
    int total_elements = 1;
    for (int i = 0; i < ndims_x; i++) {
        total_elements *= x_dims[i];
    }

    // Create output dimensions: z has shape (ndims_x, x_dims...)
    mwSize z_dims[9];  // Support up to 8 + 1 dimensions
    z_dims[0] = ndims_x;
    for (int i = 0; i < ndims_x; i++) {
        z_dims[i + 1] = x_dims[i];
    }
    int ndims_z = ndims_x + 1;

    // Create output array z
    mxGPUArray *z_result = mxGPUCreateGPUArray(ndims_z, z_dims, x_class, mxREAL, MX_GPU_INITIALIZE_VALUES);

    // Copy dimensions and compute strides
    int *d_dims, *d_strides;
    int dims_host[8], strides_host[8];

    for (int i = 0; i < ndims_x; i++) {
        dims_host[i] = (int)x_dims[i];
    }

    // Compute strides
    strides_host[0] = 1;
    for (int d = 1; d < ndims_x; d++) {
        strides_host[d] = strides_host[d-1] * dims_host[d-1];
    }

    cudaMalloc(&d_dims, ndims_x * sizeof(int));
    cudaMalloc(&d_strides, ndims_x * sizeof(int));
    cudaMemcpy(d_dims, dims_host, ndims_x * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, strides_host, ndims_x * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel based on data type
    int threadsPerBlock_sdiff_T;
    int minblocksPerGrid_sdiff_T;
    int blocksPerGrid_sdiff_T;
    int threadsPerBlock_diff_z;
    int minblocksPerGrid_diff_z;
    int blocksPerGrid_diff_z;
    if (x_class == mxDOUBLE_CLASS) {
        cudaOccupancyMaxPotentialBlockSize(&minblocksPerGrid_sdiff_T, &threadsPerBlock_sdiff_T, (void*)spatial_diff_T_kernel<double>, 0, total_elements);
        blocksPerGrid_sdiff_T = (total_elements + threadsPerBlock_sdiff_T - 1) / threadsPerBlock_sdiff_T;
        cudaOccupancyMaxPotentialBlockSize(&minblocksPerGrid_diff_z, &threadsPerBlock_diff_z, (void*)spatial_diff_T_kernel<double>, 0, total_elements);
        blocksPerGrid_diff_z = (total_elements + threadsPerBlock_diff_z - 1) / threadsPerBlock_diff_z;

        // Get pointers to data
        double *z_data = (double*)mxGPUGetData(z_result);
        const double *x_data = (const double*)mxGPUGetDataReadOnly(x_gpu);

        // Allocate temporary x_tmp array
        double *x_tmp_data;
        cudaMalloc(&x_tmp_data, total_elements * sizeof(double));

        // Main optimization loop
        for (int iter = 0; iter < niter; iter++) {
            // Step 1: x_tmp = x + w * spatial_diff_T(z)
            spatial_diff_T_kernel<double><<<blocksPerGrid_sdiff_T, threadsPerBlock_sdiff_T>>>(
                x_tmp_data, x_data, z_data, (double)w, ndims_x, d_dims, d_strides, total_elements);

            // Step 2: spatial_diff(x_tmp) and z update
            spatial_diff_z_update_kernel<double><<<blocksPerGrid_diff_z, threadsPerBlock_diff_z>>>(
                z_data, x_tmp_data, (double)v, projection_type, ndims_x, d_dims, d_strides, total_elements);
        }

        // Clean up x_tmp
        cudaFree(x_tmp_data);
    } else { // mxSINGLE_CLASS
        cudaOccupancyMaxPotentialBlockSize(&minblocksPerGrid_sdiff_T, &threadsPerBlock_sdiff_T, (void*)spatial_diff_T_kernel<float>, 0, total_elements);
        blocksPerGrid_sdiff_T = (total_elements + threadsPerBlock_sdiff_T - 1) / threadsPerBlock_sdiff_T;
        cudaOccupancyMaxPotentialBlockSize(&minblocksPerGrid_diff_z, &threadsPerBlock_diff_z, (void*)spatial_diff_T_kernel<float>, 0, total_elements);
        blocksPerGrid_diff_z = (total_elements + threadsPerBlock_diff_z - 1) / threadsPerBlock_diff_z;

        // Get pointers to data
        float *z_data = (float*)mxGPUGetData(z_result);
        const float *x_data = (const float*)mxGPUGetDataReadOnly(x_gpu);

        // Allocate temporary x_tmp array
        float *x_tmp_data;
        cudaMalloc(&x_tmp_data, total_elements * sizeof(float));

        // Main optimization loop
        for (int iter = 0; iter < niter; iter++) {
            // Step 1: x_tmp = x + w * spatial_diff_T(z)
            spatial_diff_T_kernel<float><<<blocksPerGrid_sdiff_T, threadsPerBlock_sdiff_T>>>(
                x_tmp_data, x_data, z_data, (float)w, ndims_x, d_dims, d_strides, total_elements);

            // Step 2: spatial_diff(x_tmp) and z update
            spatial_diff_z_update_kernel<float><<<blocksPerGrid_diff_z, threadsPerBlock_diff_z>>>(
                z_data, x_tmp_data, (float)v, projection_type, ndims_x, d_dims, d_strides, total_elements);
        }

        // Clean up x_tmp
        cudaFree(x_tmp_data);
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
    cudaFree(d_strides);

    // Create output
    plhs[0] = mxGPUCreateMxArrayOnGPU(z_result);

    // Clean up
    mxGPUDestroyGPUArray(x_gpu);
    mxGPUDestroyGPUArray(z_result);
}