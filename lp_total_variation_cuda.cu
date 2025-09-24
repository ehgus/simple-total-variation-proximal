#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cmath>

// Device function: spatial_diff_T operation
// Input: z_data (input array ndims, total_elements), dims, strides, ndims
// Output: returns the result
template<typename T>
__device__ T spatial_diff_T_device(const T* z_data, int idx, int ndims, const int* dims, const int* strides, const int total_elements) {
    // Calculate multi-dimensional coordinates from linear index
    int temp_idx = idx;
    int coords[4];
    for (int d = 0; d < ndims; d++) {
        coords[d] = temp_idx % dims[d];
        temp_idx /= dims[d];
    }

    // Initialize with sum across last dimension (sum of z over last dim)
    T sum_val = T(0.0);
    for (int dim = 0; dim < ndims; dim++) {
        sum_val += z_data[idx + total_elements * dim];
    }

    // Compute transpose spatial differences for each dimension
    for (int dim = 0; dim < ndims; dim++) {
        // Apply forward circular shift: move forward by 1 in current dimension
        int neighbor_coord = (coords[dim] + 1) % dims[dim];

        // Calculate neighbor index efficiently using pre-computed strides
        int neighbor_idx = idx + (neighbor_coord - coords[dim]) * strides[dim];

        // Subtract the shifted value (transpose operation)
        sum_val -= z_data[neighbor_idx + total_elements * dim];
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
    int coords[4];
    for (int d = 0; d < ndims; d++) {
        coords[d] = temp_idx % dims[d];
        temp_idx /= dims[d];
    }

    // Compute spatial differences for each dimension
    for (int dim = 0; dim < ndims; dim++) {
        // Apply circular shift: move backward by 1 in current dimension
        int neighbor_coord = (coords[dim] - 1 + dims[dim]) % dims[dim];

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
    norm_val = sqrt(norm_val);

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
    T spatial_diff_T_result = spatial_diff_T_device(z_data, idx, ndims, dims, strides, total_elements);
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
    T z_tmp_local[4];  // Support up to 4 dimensions
    T spatial_diff_result[4];

    // Step 2: z_tmp = z - v * spatial_diff(x_tmp)
    spatial_diff_device(x_tmp_data, spatial_diff_result, idx, ndims, dims, strides);
    for (int d = 0; d < ndims; d++) {
        z_tmp_local[d] = z_data[idx + total_elements * d] - v * spatial_diff_result[d];
    }

    // Step 3: z = v * norm.projection(z, z_tmp / v)
    T z_tmp_scaled[4];
    for (int d = 0; d < ndims; d++) {
        z_tmp_scaled[d] = z_tmp_local[d] / v;
    }

    T z_projected[4];
    unit_ball_projection_device(z_projected, z_tmp_scaled, ndims, projection_type);

    for (int d = 0; d < ndims; d++) {
        z_data[idx + total_elements * d] = v * z_projected[d];
    }
}

// Template function for the optimization algorithm
template<typename T>
void lp_total_variation_optimize(
    const T* x_data,            // Input data
    T* y_data,                  // Output data
    T w, T v,                   // Weight parameters
    int niter,                  // Number of iterations
    int projection_type,        // Projection type
    int ndims_x,                // Number of dimensions
    int total_elements,         // Total elements
    int z_total_elements,       // Total z elements
    const int* d_dims,          // Device dimensions
    const int* d_strides        // Device strides
) {
    // Calculate optimal block sizes
    int threadsPerBlock_sdiff_T, minblocksPerGrid_sdiff_T, blocksPerGrid_sdiff_T;
    int threadsPerBlock_diff_z, minblocksPerGrid_diff_z, blocksPerGrid_diff_z;

    cudaOccupancyMaxPotentialBlockSize(&minblocksPerGrid_sdiff_T, &threadsPerBlock_sdiff_T,
        (void*)spatial_diff_T_kernel<T>, 0, total_elements);
    blocksPerGrid_sdiff_T = (total_elements + threadsPerBlock_sdiff_T - 1) / threadsPerBlock_sdiff_T;

    cudaOccupancyMaxPotentialBlockSize(&minblocksPerGrid_diff_z, &threadsPerBlock_diff_z,
        (void*)spatial_diff_z_update_kernel<T>, 0, total_elements);
    blocksPerGrid_diff_z = (total_elements + threadsPerBlock_diff_z - 1) / threadsPerBlock_diff_z;

    // Allocate native CUDA memory for z array (optimization variable)
    T *z_data;
    cudaMalloc(&z_data, z_total_elements * sizeof(T));
    cudaMemset(z_data, 0, z_total_elements * sizeof(T)); // Initialize to zero

    // Main optimization loop
    for (int iter = 0; iter < niter; iter++) {
        // Step 1: spatial_diff(y) and z update
        spatial_diff_z_update_kernel<T><<<blocksPerGrid_diff_z, threadsPerBlock_diff_z>>>(
            z_data, y_data, v, projection_type, ndims_x, d_dims, d_strides, total_elements);

        // Step 2: y = x + w * spatial_diff_T(z)
        spatial_diff_T_kernel<T><<<blocksPerGrid_sdiff_T, threadsPerBlock_sdiff_T>>>(
            y_data, x_data, z_data, w, ndims_x, d_dims, d_strides, total_elements);
    }

    // Wait for completion
    cudaDeviceSynchronize();

    // Check for kernel errors
    if (cudaGetLastError() != cudaSuccess) {
        mexErrMsgTxt("CUDA kernel execution failed");
    }

    // Clean up native CUDA arrays
    cudaFree(z_data);
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

    if (ndims_x > 4) {
        mexErrMsgTxt("Maximum 4 dimensions supported");
    }

    // Calculate total elements
    int total_elements = 1;
    for (int i = 0; i < ndims_x; i++) {
        total_elements *= x_dims[i];
    }

    // Calculate z array size: total_elements * ndims_x
    int z_total_elements = total_elements * ndims_x;

    // Create output array y for final result (same size as input x)
    mxGPUArray *y_result = mxGPUCopyFromMxArray(prhs[0]);

    // Copy dimensions and compute strides
    int *d_dims, *d_strides;
    int dims_host[4], strides_host[4];

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
    if (x_class == mxDOUBLE_CLASS) {
        // Get pointers to data
        const double *x_data = (const double*)mxGPUGetDataReadOnly(x_gpu);
        double *y_data = (double*)mxGPUGetData(y_result);

        // Run optimization algorithm
        lp_total_variation_optimize<double>(
            x_data, y_data, (double)w, (double)v, niter, projection_type,
            ndims_x, total_elements, z_total_elements, d_dims, d_strides);
    } else { // mxSINGLE_CLASS
        // Get pointers to data
        const float *x_data = (const float*)mxGPUGetDataReadOnly(x_gpu);
        float *y_data = (float*)mxGPUGetData(y_result);

        // Run optimization algorithm
        lp_total_variation_optimize<float>(
            x_data, y_data, (float)w, (float)v, niter, projection_type,
            ndims_x, total_elements, z_total_elements, d_dims, d_strides);
    }

    // Clean up device memory
    cudaFree(d_dims);
    cudaFree(d_strides);

    // Create output
    plhs[0] = mxGPUCreateMxArrayOnGPU(y_result);

    // Clean up
    mxGPUDestroyGPUArray(x_gpu);
    mxGPUDestroyGPUArray(y_result);
}