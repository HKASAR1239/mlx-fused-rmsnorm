#include <cmath>
#include <limits>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "mlx/fast.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

mx::array residual_rms_norm(
    const mx::array& x,
    const mx::array& residual,
    const mx::array& weight,
    float eps,
    mx::StreamOrDevice stream = {}) {
  if (x.shape() != residual.shape() || x.ndim() < 1 || x.size() == 0) {
    throw std::invalid_argument("x and residual need the same nonempty shape");
  }

  int width = x.shape().back();
  if (weight.ndim() != 1 || weight.size() != width) {
    throw std::invalid_argument("weight must have shape (x.shape[-1],)");
  }
  if (x.dtype() != residual.dtype() || x.dtype() != weight.dtype() ||
      (x.dtype() != mx::float16 && x.dtype() != mx::bfloat16 &&
       x.dtype() != mx::float32)) {
    throw std::invalid_argument("all inputs must share float16, bfloat16 or float32 dtype");
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    throw std::invalid_argument("eps must be positive and finite");
  }

  auto rows = x.size() / width;
  if (rows > std::numeric_limits<int>::max() / 32) {
    throw std::invalid_argument("too many rows");
  }

  static const auto kernel = mx::fast::metal_kernel(
      "residual_rms_norm",
      {"x", "residual", "weight", "eps"},
      {"out"},
      R"metal(
        uint row = thread_position_in_grid.x / 32;
        uint lane = thread_position_in_grid.x % 32;
        uint width = x_shape[x_ndim - 1];
        float sum = 0.0f;

        for (uint col = lane; col < width; col += 32) {
          uint index = row * width + col;
          T summed = x[index] + residual[index];
          float value = float(summed);
          sum += value * value;
        }

        float scale = rsqrt(simd_sum(sum) / float(width) + eps);
        for (uint col = lane; col < width; col += 32) {
          uint index = row * width + col;
          T summed = x[index] + residual[index];
          float value = float(summed);
          out[index] = T(value * scale * float(weight[col]));
        }
      )metal");

  return kernel(
             {x, residual, weight, mx::array(eps)},
             {x.shape()},
             {x.dtype()},
             {static_cast<int>(rows * 32), 1, 1},
             {32, 1, 1},
             {{"T", x.dtype()}},
             std::nullopt,
             false,
             stream)
      .front();
}

NB_MODULE(_ext, m) {
  m.def(
      "residual_rms_norm",
      &residual_rms_norm,
      "x"_a,
      "residual"_a,
      "weight"_a,
      "eps"_a = 1e-5f,
      nb::kw_only(),
      "stream"_a = nb::none());
}
