# Fused residual RMSNorm for MLX

An inference-only Metal kernel for `RMSNorm(x + residual, weight)`, exposed through a small C++/Python extension. Each row uses one threadgroup, with up to eight SIMD groups reducing in float32. The baseline is `mlx.core.fast.rms_norm(x + residual, weight, eps)`.

Requires an Apple Silicon Mac running macOS 15 or newer, Xcode with the Metal Toolchain, and Python 3.10 or newer. The commands below use Python 3.12.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/python bench.py > results.csv
```

```python
from mlx_fused_rmsnorm import residual_rms_norm

y = residual_rms_norm(x, residual, weight, eps=1e-5)
```

The kernel accepts float16, bfloat16, and float32 inputs, including noncontiguous arrays. The three inputs must have the same dtype; `weight` must match the final dimension of `x`. It does not implement gradients or batching transforms.

`bench.py` writes median time per call after warmup to CSV and prints the MLX version and GPU to stderr. A speedup is not assumed: the result depends on shape, dtype, and Apple Silicon generation.

For a more useful comparison, run `bench_v2.py`. It checks numerical agreement, compares eager MLX, `mx.compile` and the extension, and reports both synchronized calls (`serial`) and groups of 16 independent calls evaluated together (`batch`). It rotates measurement order and reports median absolute deviation (`mad_us`). Both modes include Python and MLX scheduling overhead; neither is a GPU-only kernel timing.

```sh
.venv/bin/python bench_v2.py > results-v2.csv
MTL_CAPTURE_ENABLED=1 .venv/bin/python profile.py mlx
MTL_CAPTURE_ENABLED=1 .venv/bin/python profile.py compiled
MTL_CAPTURE_ENABLED=1 .venv/bin/python profile.py fused
```

Open the `.gputrace` files in Xcode's Metal debugger to inspect GPU kernel durations and dispatch counts. Profile the same shape and dtype for each variant (`--rows`, `--width`, `--dtype`); the defaults are 128 × 4096 float16. Trace files are local and excluded from Git.

On an Apple M4 Max with MLX 0.32.2, all 13 tests pass. The [initial synchronized benchmark](benchmarks/m4-max-mlx-0.32.2.csv) ranges from 0.90× to 1.20× against eager MLX. In the [controlled benchmark](benchmarks/m4-max-mlx-0.32.2-v2.csv), the 128 × 4096 float32 case takes 38.15 µs per call with eager MLX, 37.59 µs with compiled MLX, and 29.82 µs with this extension in batch mode (1.28× versus eager MLX). The float16 and bfloat16 cases at that shape reach 1.11× and 1.13×; smaller shapes are usually slower with the extension. Batch mode reduces the effect of per-call synchronization but still includes Python and scheduling overhead. GPU-only timing requires the Metal traces. MLX and nanobind are pinned to compatible versions because their C++ array bindings share an ABI.

Based on the [MLX custom Metal kernel](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) and [extension](https://ml-explore.github.io/mlx/build/html/dev/extensions.html) APIs.
