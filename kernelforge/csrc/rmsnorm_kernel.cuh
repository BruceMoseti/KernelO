// Device code for the handwritten CUDA RMSNorm.
//
// Separated from the ATen launch site in rmsnorm.cu for the usual reason a
// templated CUDA kernel lives in a header, and for one specific one: this
// translation unit depends only on the CUDA runtime, so it can be compiled to
// PTX on a machine with no GPU and no nvcc (clang in CUDA mode is enough).
// tests/test_cuda_rmsnorm.py does exactly that, which keeps the device code
// under CI instead of unverified until someone runs it.

#pragma once

#include <cuda_runtime.h>

namespace kernelforge {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 1024;

// Sum a value across one warp. __shfl_down_sync exchanges registers directly,
// so no shared memory is needed until the per-warp partials are combined.
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// One block per row. Pass one reduces the sum of squares, pass two rescales.
//
// This mirrors the Triton kernel's strategy so that the comparison measures
// the implementations rather than two different algorithms, with one
// structural difference worth knowing when reading the numbers: Triton holds
// the row in registers across both passes and touches global memory once,
// while this strides over the row twice and serves the second pass from
// L1/L2. Whether that costs anything depends on the row width against the
// cache, which is what the profiler comparison is for.
template <typename scalar_t>
__global__ void rmsnorm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gamma,
    scalar_t* __restrict__ y,
    const int n_cols,
    const long long row_stride,
    const float eps) {
  // One float per warp to combine the partials, then slot 0 is reused to
  // broadcast the reciprocal RMS to the whole block.
  extern __shared__ float shared[];

  const long long row = blockIdx.x;
  const scalar_t* x_row = x + row * row_stride;
  scalar_t* y_row = y + row * row_stride;

  float sum_squares = 0.0f;
  for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
    // Accumulate in fp32 whatever the storage type: in fp16 a running sum over
    // a 4096-wide row of unit-variance values reaches ~4096, where 11 bits of
    // mantissa stop resolving the individual terms.
    const float value = static_cast<float>(x_row[col]);
    sum_squares += value * value;
  }

  sum_squares = warp_reduce_sum(sum_squares);

  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  if (lane == 0) {
    shared[warp] = sum_squares;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    float total = 0.0f;
    for (int i = 0; i < num_warps; ++i) {
      total += shared[i];
    }
    shared[0] = rsqrtf(total / static_cast<float>(n_cols) + eps);
  }
  __syncthreads();

  const float inv_rms = shared[0];
  for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
    const float value =
        static_cast<float>(x_row[col]) * inv_rms * static_cast<float>(gamma[col]);
    y_row[col] = static_cast<scalar_t>(value);
  }
}

}  // namespace kernelforge
