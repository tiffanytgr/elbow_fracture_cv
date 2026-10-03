#!/usr/bin/env bash
#
# End-to-end smoke test on a synthetic dataset: manifest -> train -> evaluate
# -> compare -> figures. It verifies that the code runs and that the artefacts
# connect to each other, not that the model learns anything -- the dataset is
# 45 random-noise images.
#
# Run it after setting up a machine, before committing hours of GPU time:
#
#     bash research/smoke_test.sh [output_dir]
#
# Takes about a minute on CPU. Prints SMOKE TEST PASSED on success.
#
set -euo pipefail

ROOT="${1:-$(mktemp -d -t elbow-smoke-XXXXXX)}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p "$ROOT"
echo "working in $ROOT"

python3 - "$ROOT" <<'PY'
"""Write a tiny fake dataset in the KKH folder layout, plus per-view manifests."""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
data = root / "data"
rng = np.random.default_rng(0)
folders = ["Normal", "SC Grade 1", "SC Grade 2a", "SC Grade 2b", "SC Grade 3"]

case = 0
rows: dict[str, list[tuple[str, str, str]]] = {"train": [], "val": [], "test": []}
for grade_index, folder in enumerate(folders):
    for i in range(9):
        case += 1
        case_id = f"A{case:04d}"
        split = "train" if i < 5 else ("val" if i < 7 else "test")
        for view in ("AP", "LAT"):
            directory = data / folder / f"{folder} {view}"
            directory.mkdir(parents=True, exist_ok=True)
            # Brightness varies with grade so the loss has something to move on.
            pixels = rng.normal(40 + 35 * grade_index, 18, (96, 96))
            path = directory / f"{case_id} {view}.png"
            Image.fromarray(np.clip(pixels, 0, 255).astype("uint8")).save(path)
            rows[split].append((str(path), folder, view))

for split, entries in rows.items():
    for view in ("AP", "LAT"):
        name = f"{'' if view == 'AP' else 'LAT_'}{split}.csv"
        with (root / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Filepath", "Label"])
            writer.writerows(
                (path, label) for path, label, v in entries if v == view
            )
print(f"wrote {case} synthetic cases")
PY

echo "=== manifest ==="
python3 -m research.multiclass_baseline.manifest \
  --ap-csv  "train=$ROOT/train.csv" --lat-csv "train=$ROOT/LAT_train.csv" \
  --ap-csv  "val=$ROOT/val.csv"     --lat-csv "val=$ROOT/LAT_val.csv" \
  --ap-csv  "test=$ROOT/test.csv"   --lat-csv "test=$ROOT/LAT_test.csv" \
  --out "$ROOT/paired_manifest.csv"

# --no-imagenet keeps the test offline. Real runs omit it, or pass
# --encoder-checkpoint to start from the project's autoencoder encoder.
for arch in dual two-channel; do
  echo "=== train ($arch) ==="
  python3 -m research.multiclass_baseline.train \
    --manifest "$ROOT/paired_manifest.csv" --architecture "$arch" \
    --out-dir "$ROOT/run_$arch" --seed 0 --device cpu --no-imagenet \
    --stage1-epochs 1 --stage2-epochs 1 --batch-size 8 --num-workers 0
done

# A retained-case list drawn from the test split, to exercise --coverage-from.
awk -F, '$2=="test"{print $1}' "$ROOT/paired_manifest.csv" | head -7 > "$ROOT/retained.txt"

echo "=== evaluate (dual, with coverage filtering) ==="
python3 -m research.multiclass_baseline.evaluate \
  --manifest "$ROOT/paired_manifest.csv" \
  --checkpoint "$ROOT/run_dual/best_model.pth" \
  --split test --out-dir "$ROOT/eval_dual" --device cpu \
  --num-workers 0 --n-boot 100 --coverage-from "$ROOT/retained.txt"

echo "=== evaluate (two-channel) ==="
python3 -m research.multiclass_baseline.evaluate \
  --manifest "$ROOT/paired_manifest.csv" \
  --checkpoint "$ROOT/run_two-channel/best_model.pth" \
  --split test --out-dir "$ROOT/eval_2ch" --device cpu \
  --num-workers 0 --n-boot 100

echo "=== compare the two arms ==="
python3 -m research.multiclass_baseline.compare \
  --cascade "$ROOT/eval_dual/predictions.csv" \
  --multiclass "$ROOT/eval_2ch/predictions.csv" \
  --out "$ROOT/comparison.json" --n-boot 100

echo "=== figures from the evaluation output ==="
mkdir -p "$ROOT/results"
cp "$ROOT/eval_dual/predictions.csv" "$ROOT/results/cascade_predictions.csv"
cp "$ROOT/eval_2ch/predictions.csv"  "$ROOT/results/multiclass_predictions.csv"
cp "$ROOT/run_dual/training_curve.csv" "$ROOT/results/training_curve.csv"
python3 -m research.figures.fig2_classification \
  --results "$ROOT/results" --out-dir "$ROOT/figures" --formats png
python3 -m research.figures.appendix_figures \
  --results "$ROOT/results" --out-dir "$ROOT/figures" --formats png --only s2

echo
echo "SMOKE TEST PASSED -- artefacts in $ROOT"
