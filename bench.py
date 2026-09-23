import argparse
import platform
import statistics
import time

import mlx.core as mx

from mlx_fused_rmsnorm import residual_rms_norm


def run(fn, repeats, iterations):
    for _ in range(10):
        mx.eval(fn())
    mx.synchronize()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    device = mx.metal.device_info().get("device_name", "Apple GPU")
    print(f"MLX {mx.__version__} | {device} | {platform.platform()}")
    print("rows,width,dtype,mlx_us,fused_us,speedup")
    for rows, width in [(1, 256), (1, 4096), (32, 1024), (128, 4096)]:
        for dtype in (mx.float32, mx.float16):
            x = mx.random.normal((rows, width)).astype(dtype)
            residual = mx.random.normal((rows, width)).astype(dtype)
            weight = mx.random.normal((width,)).astype(dtype)
            mx.eval(x, residual, weight)

            baseline = lambda: mx.fast.rms_norm(x + residual, weight, 1e-5)
            fused = lambda: residual_rms_norm(x, residual, weight)
            mlx_us = run(baseline, args.repeats, args.iterations)
            fused_us = run(fused, args.repeats, args.iterations)
            print(
                f"{rows},{width},{dtype},{mlx_us:.2f},{fused_us:.2f},"
                f"{mlx_us / fused_us:.2f}"
            )


if __name__ == "__main__":
    main()
