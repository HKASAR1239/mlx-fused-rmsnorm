# Fused residual RMSNorm for MLX

An inference-only Metal kernel for `RMSNorm(x + residual, weight)`, exposed through a small C++/Python extension. One SIMD group processes each row and accumulates in float32. The baseline is `mlx.core.fast.rms_norm(x + residual, weight, eps)`.

Requires an Apple Silicon Mac running macOS 15 or newer, Xcode with the Metal Toolchain, and Python 3.10 or newer. The commands below use Python 3.12.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/python bench.py > results.csv
```

The kernel accepts float16 and float32 inputs, including noncontiguous arrays. The three inputs must have the same dtype; `weight` must match the final dimension of `x`. It does not implement gradients or batching transforms.

`bench.py` writes median time per call after warmup to CSV and prints the MLX version and GPU to stderr. A speedup is not assumed: the result depends on shape, dtype, and Apple Silicon generation.

Based on the [MLX custom Metal kernel](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) and [extension](https://ml-explore.github.io/mlx/build/html/dev/extensions.html) APIs.
