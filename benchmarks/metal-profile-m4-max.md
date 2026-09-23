# Metal capture notes: M4 Max

Captured on an Apple M4 Max with macOS 15.7.4, MLX 0.32.2 and Xcode 26.2. Command:

```sh
for variant in mlx compiled fused; do
  MTL_CAPTURE_ENABLED=1 .venv/bin/python profile.py "$variant" --rows 128 --width 4096 --dtype float32
done
```

Xcode's replay of each serial capture showed 19 compute encoders. Eager MLX and `mx.compile` each had 38 dispatches (an addition shader and an RMSNorm shader per encoder); the extension had 19 dispatches (one custom shader per encoder). `mx.compile` did not combine the two MLX shaders for this workload.

The replay reported effective GPU times of 284.49 µs for eager MLX, 269.79 µs for compiled MLX and 325.94 µs for the extension, all at Xcode's “Medium” performance state. Individual encoder costs varied substantially. These serial replay totals do not establish a GPU-time win and should not be equated with the batch wall-clock medians in the CSV. The batch-mode captures are intended to reduce idle gaps and make the shader comparison more informative.
