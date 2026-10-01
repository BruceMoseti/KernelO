// ATen launch site for the handwritten CUDA RMSNorm.
//
// The point of writing this operator twice is not that C++ must be faster. If
// Triton wins, that is a finding about what the compiler does with the same
// strategy; if CUDA wins, that is a finding about what Triton gives up. Either
// way the comparison only means something if both sides implement the same
// algorithm, which is why the device code in rmsnorm_kernel.cuh is a
// deliberate transcription of the Triton kernel rather than a better one.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

#include "rmsnorm_kernel.cuh"

at::Tensor rmsnorm_forward_cuda(const at::Tensor& x, const at::Tensor& gamma, double eps) {
  const at::Tensor x_c = x.contiguous();
  const at::Tensor gamma_c = gamma.contiguous();

  const int n_rows = static_cast<int>(x_c.size(0));
  const int n_cols = static_cast<int>(x_c.size(1));
  at::Tensor y = at::empty_like(x_c);

  if (n_rows == 0 || n_cols == 0) {
    return y;
  }

  // Enough threads to cover the row, rounded up to a whole warp and capped at
  // the block limit; wider rows are handled by the strided loop in the kernel.
  const int threads = std::min(
      kernelforge::kMaxThreads,
      ((n_cols + kernelforge::kWarpSize - 1) / kernelforge::kWarpSize) * kernelforge::kWarpSize);
  const int num_warps =
      (threads + kernelforge::kWarpSize - 1) / kernelforge::kWarpSize;
  const size_t shared_bytes = static_cast<size_t>(num_warps) * sizeof(float);

  const c10::cuda::CUDAGuard guard(x_c.device());
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x_c.scalar_type(),
      "rmsnorm_forward_cuda", [&] {
        kernelforge::rmsnorm_forward_kernel<scalar_t>
            <<<n_rows, threads, shared_bytes, stream>>>(
                x_c.const_data_ptr<scalar_t>(),
                gamma_c.const_data_ptr<scalar_t>(),
                y.mutable_data_ptr<scalar_t>(),
                n_cols,
                static_cast<long long>(n_cols),
                static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
