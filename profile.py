import argparse
import os
from pathlib import Path

import mlx.core as mx

from bench_v2 import mlx_norm
from mlx_fused_rmsnorm import residual_rms_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("mlx", "compiled", "fused"))
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mode", choices=("serial", "batch"), default="serial")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("MTL_CAPTURE_ENABLED"):
        parser.error("set MTL_CAPTURE_ENABLED=1 before launching Python")
    if min(args.rows, args.width, args.iterations) < 1:
        parser.error("rows, width and iterations must be positive")

    dtype = getattr(mx, args.dtype)
    inputs = [
        (
            mx.random.normal((args.rows, args.width)).astype(dtype),
            mx.random.normal((args.rows, args.width)).astype(dtype),
            mx.random.normal((args.width,)).astype(dtype),
        )
        for _ in range(args.iterations if args.mode == "batch" else 1)
    ]
    mx.eval(*(array for item in inputs for array in item))
    fn = {
        "mlx": mlx_norm,
        "compiled": mx.compile(mlx_norm),
        "fused": residual_rms_norm,
    }[args.variant]
    for _ in range(10):
        mx.eval(fn(*inputs[0]))
    mx.synchronize()

    output = args.output or Path(
        f"{args.variant}-{'batch-' if args.mode == 'batch' else ''}"
        f"{args.rows}x{args.width}-{args.dtype}.gputrace"
    )
    mx.metal.start_capture(str(output.resolve()))
    try:
        if args.mode == "batch":
            mx.eval(*(fn(*item) for item in inputs))
        else:
            for _ in range(args.iterations):
                mx.eval(fn(*inputs[0]))
        mx.synchronize()
    finally:
        mx.metal.stop_capture()
    print(output.resolve())


if __name__ == "__main__":
    main()
