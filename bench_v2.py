import argparse
import platform
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

from mlx_fused_rmsnorm import residual_rms_norm


def mlx_norm(x, residual, weight):
    return mx.fast.rms_norm(x + residual, weight, 1e-5)


def measure(fn, inputs, iterations, batched):
    start = time.perf_counter()
    for _ in range(iterations):
        if batched:
            mx.eval(*(fn(*item) for item in inputs))
        else:
            mx.eval(fn(*inputs[0]))
    mx.synchronize()
    calls = iterations * (len(inputs) if batched else 1)
    return (time.perf_counter() - start) * 1e6 / calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--case", action="append", metavar="ROWSxWIDTH")
    parser.add_argument("--dtype", action="append", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--mode", choices=("serial", "batch", "both"), default="both")
    args = parser.parse_args()
    if min(args.repeats, args.iterations, args.batch_size) < 1:
        parser.error("repeats, iterations and batch-size must be positive")
    cases = []
    for case in args.case or ("1x256", "1x4096", "32x1024", "128x4096"):
        try:
            rows, width = map(int, case.split("x"))
        except ValueError:
            parser.error(f"invalid case: {case}")
        if min(rows, width) < 1:
            parser.error(f"invalid case: {case}")
        cases.append((rows, width))

    device = mx.device_info(mx.gpu).get("device_name", "Apple GPU")
    print(f"MLX {mx.__version__} | {device} | {platform.platform()}", file=sys.stderr)
    mx.random.seed(0)
    print("mode,rows,width,dtype,variant,us_per_call,mad_us,speedup_vs_mlx")
    variants = {"mlx": mlx_norm, "compiled": mx.compile(mlx_norm), "fused": residual_rms_norm}
    for rows, width in cases:
        for name in args.dtype or ("float32", "float16", "bfloat16"):
            dtype = getattr(mx, name)
            inputs = [
                (
                    mx.random.normal((rows, width)).astype(dtype),
                    mx.random.normal((rows, width)).astype(dtype),
                    mx.random.normal((width,)).astype(dtype),
                )
                for _ in range(args.batch_size)
            ]
            mx.eval(*(array for item in inputs for array in item))
            expected = mlx_norm(*inputs[0])
            tolerance = {"float32": 2e-5, "float16": 2e-2, "bfloat16": 3e-2}[name]
            for variant in ("compiled", "fused"):
                actual = variants[variant](*inputs[0])
                mx.eval(expected, actual)
                np.testing.assert_allclose(
                    np.array(actual.astype(mx.float32)),
                    np.array(expected.astype(mx.float32)),
                    rtol=tolerance,
                    atol=tolerance,
                )

            for mode in (("serial", "batch") if args.mode == "both" else (args.mode,)):
                for fn in variants.values():
                    for _ in range(5):
                        mx.eval(fn(*inputs[0]))
                samples = {variant: [] for variant in variants}
                names = list(variants)
                for repeat in range(args.repeats):
                    for variant in names[repeat % len(names):] + names[:repeat % len(names)]:
                        samples[variant].append(
                            measure(variants[variant], inputs, args.iterations, mode == "batch")
                        )
                results = {variant: statistics.median(values) for variant, values in samples.items()}
                for variant, duration in results.items():
                    mad = statistics.median(abs(value - duration) for value in samples[variant])
                    print(
                        f"{mode},{rows},{width},{name},{variant},"
                        f"{duration:.2f},{mad:.2f},{results['mlx'] / duration:.2f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
