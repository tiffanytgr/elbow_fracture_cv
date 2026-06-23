"""Smoke test for the pipeline package.

Picks one image from each grade folder under experiments/aligned_images/,
runs predict() on AP, LAT, and AP+LAT combinations, and prints results.

Run:
    cd D:\\OneDrive - Nanyang Technological University\\AI6129\\KKH_Elbow
    python -m pipeline.test_pipeline
"""
from __future__ import annotations
import json
from pathlib import Path

from pipeline import predict, default_config


ALIGNED_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "aligned_images"

LAT_FOLDERS = ["SC Grade 1 LAT", "SC Grade 2a LAT", "SC Grade 2b LAT", "SC Grade 3 LAT"]
AP_FOLDERS  = ["SC Grade 1 AP",  "SC Grade 2a AP",  "SC Grade 2b AP",  "SC Grade 3 AP"]


def _first_image(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        for p in sorted(folder.glob(f"*{ext}")):
            return p
    return None


def _summary(result, header: str):
    print(f"\n{'='*70}\n{header}\n{'='*70}")
    print(f"Final grade: {result.final_grade}  (source={result.grade_source}, discordant={result.discordant})")
    print(f"CNN grade  : {result.cnn_final_grade}")
    if result.exp1:
        print(f"  Exp1 (Normal/Fractured): {result.exp1.labels[result.exp1.pred_idx] if result.exp1.pred_idx >= 0 else 'SKIPPED'}"
              f" reason={result.exp1.skipped_reason or '-'} ood={result.exp1.ood_score}")
    if result.exp2:
        print(f"  Exp2 (G3 vs G1/2)      : {result.exp2.labels[result.exp2.pred_idx] if result.exp2.pred_idx >= 0 else 'SKIPPED'}"
              f" reason={result.exp2.skipped_reason or '-'}")
    if result.exp3:
        print(f"  Exp3 (G1 vs G2)        : {result.exp3.labels[result.exp3.pred_idx]} "
              f"probs={[f'{p:.3f}' for p in result.exp3.probs]}")
    if result.exp4:
        if result.exp4.pred_idx >= 0:
            print(f"  Exp4 (G2a vs G2b)      : {result.exp4.labels[result.exp4.pred_idx]} "
                  f"probs={[f'{p:.3f}' for p in result.exp4.probs]}")
        else:
            print(f"  Exp4                   : SKIPPED ({result.exp4.skipped_reason})")
    if result.geometric:
        g = result.geometric
        print(f"Geometric  : {g.final_grade} (1v2={g.grade_1v2} 2ab={g.grade_2ab})")
        if g.skipped_reason:
            print(f"  geometric skipped: {g.skipped_reason}")
        if g.ahl_diagnostic:
            print(f"  AHL: a={g.ahl_diagnostic['ahl_a']:.4f} b={g.ahl_diagnostic['ahl_b']:.2f} "
                  f"zone={g.ahl_diagnostic['zone']} method={g.ahl_diagnostic['method']}")
        if g.width_profile:
            print(f"  width: lower={g.width_profile['lower_mean']:.1f} upper={g.width_profile['upper_mean']:.1f} "
                  f"match_ratio={g.width_profile['match_ratio']:.3f}")
    if result.log:
        print("Log:")
        for line in result.log:
            print(f"  - {line}")


def main():
    cfg = default_config()
    # For aligned_images/, we don't need the full alignment again
    cfg.run_full_alignment = False

    print(f"ALIGNED_ROOT: {ALIGNED_ROOT}")
    print(f"Exists: {ALIGNED_ROOT.exists()}")

    # Test 1: LAT-only on a Grade 2a image
    lat = _first_image(ALIGNED_ROOT / "SC Grade 2a LAT")
    if lat:
        result = predict(lat_path=lat, config=cfg)
        _summary(result, f"TEST 1: LAT only — {lat.name}")

    # Test 2: AP-only on a Grade 3 image (should trigger Grade 3 in exp2)
    ap = _first_image(ALIGNED_ROOT / "SC Grade 3 AP")
    if ap:
        result = predict(ap_path=ap, config=cfg)
        _summary(result, f"TEST 2: AP only — {ap.name}")

    # Test 3: Both views, matching same patient if possible
    lat = _first_image(ALIGNED_ROOT / "SC Grade 2b LAT")
    if lat:
        # Try to find matching AP by filename
        ap_match = None
        for ap_folder in AP_FOLDERS:
            cand = ALIGNED_ROOT / ap_folder / lat.name
            if cand.exists():
                ap_match = cand
                break
        if ap_match:
            result = predict(ap_path=ap_match, lat_path=lat, config=cfg)
            _summary(result, f"TEST 3: AP+LAT — {lat.name}")
        else:
            print(f"\nNo matching AP for {lat.name}, skipping AP+LAT test.")


if __name__ == "__main__":
    main()
