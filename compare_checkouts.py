import csv
import io
import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parent
checkouts = {
    "base": root.parent / "mlx-fused-rmsnorm",
    "tuned": root,
}
order = ("base", "tuned", "tuned", "base")
writer = None

for run, label in enumerate(order, 1):
    repo = checkouts[label]
    print(f"Run {run}/4: {label}", file=sys.stderr, flush=True)
    command = [
        str(repo / ".venv/bin/python"),
        "bench_v2.py",
        "--mode", "batch",
        "--case", "1x256",
        "--case", "128x4096",
        "--repeats", "5",
        "--iterations", "20",
    ]
    result = subprocess.run(command, cwd=repo, text=True, capture_output=True)
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    if len(rows) != 18:
        raise RuntimeError(f"expected 18 benchmark rows, got {len(rows)}")
    if writer is None:
        writer = csv.DictWriter(sys.stdout, ("run", "checkout", *rows[0]))
        writer.writeheader()
    for row in rows:
        writer.writerow({"run": run, "checkout": label, **row})
    sys.stdout.flush()
