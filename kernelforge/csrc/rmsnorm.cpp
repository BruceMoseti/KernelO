// Python bindings for the handwritten CUDA RMSNorm.
//
// Kept separate from the .cu so that pybind11 is compiled by the host
// compiler rather than by nvcc.

#include <torch/extension.h>

at::Tensor rmsnorm_forward_cuda(const at::Tensor& x, const at::Tensor& gamma, double eps);

namespace {

at::Tensor rmsnorm_forward(const at::Tensor& x, const at::Tensor& gamma, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "x must be 2D, got ", x.dim(), " dimensions");
  TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D, got ", gamma.dim(), " dimensions");
  TORCH_CHECK(
      gamma.size(0) == x.size(1),
      "gamma must have ", x.size(1), " elements, got ", gamma.size(0));
  TORCH_CHECK(
      x.scalar_type() == gamma.scalar_type(),
      "x and gamma must share a dtype, got ", x.scalar_type(), " and ", gamma.scalar_type());
  // The kernel accumulates in float and rescales with rsqrtf, so a double
  // input would be a silent precision downgrade rather than a double-precision
  // RMSNorm. Refused rather than quietly accepted.
  TORCH_CHECK(
      x.scalar_type() == at::ScalarType::Half ||
          x.scalar_type() == at::ScalarType::BFloat16 ||
          x.scalar_type() == at::ScalarType::Float,
      "unsupported dtype ", x.scalar_type(), "; expected Half, BFloat16 or Float");
  TORCH_CHECK(eps > 0.0, "eps must be positive, got ", eps);
  return rmsnorm_forward_cuda(x, gamma, eps);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "rmsnorm_forward",
      &rmsnorm_forward,
      "RMSNorm forward (CUDA)",
      pybind11::arg("x"),
      pybind11::arg("gamma"),
      pybind11::arg("eps"));
}
