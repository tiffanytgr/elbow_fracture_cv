"""CLI entry point: elbow-grader predict [options]

Install the package first:
    pip install -e "path/to/KKH_Elbow"

Then run:
    elbow-grader predict --ap AP.png --lat LAT.png
    elbow-grader predict --lat LAT.png --no-alignment --save-plots --output ./results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="elbow-grader",
        description="KKH paediatric elbow fracture grading pipeline (Gartland classification)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="Grade a single AP/LAT case")
    p.add_argument("--ap",  metavar="PATH", help="AP X-ray image path")
    p.add_argument("--lat", metavar="PATH", help="LAT X-ray image path")
    p.add_argument(
        "--checkpoint-dir", metavar="DIR",
        help=(
            "Directory with experiment checkpoint subdirs "
            "(exp1/, exp2/, exp3/, exp4/, cap_regressor/). "
            "If omitted, uses the pipeline's built-in default paths."
        ),
    )
    p.add_argument("--sam2-ckpt", metavar="PATH",
                   help="Explicit path to sam2_hiera_large.pt")
    p.add_argument("--confidence", type=float, default=0.5, metavar="FLOAT",
                   help="Confidence threshold 0.5–0.95 (default: 0.5)")
    p.add_argument("--no-alignment", action="store_true",
                   help="Skip LAT alignment (use when images are pre-aligned)")
    p.add_argument("--no-sam2", action="store_true",
                   help="Skip SAM2 geometric track (CNN-only, faster)")
    p.add_argument("--device", default="auto", metavar="DEVICE",
                   help="Compute device: auto | cpu | cuda  (default: auto)")
    p.add_argument("--output", metavar="DIR", default=".",
                   help="Directory to write outputs (default: current dir)")
    p.add_argument("--save-plots", action="store_true",
                   help="Save visualisation PNGs to --output directory")
    p.add_argument("--json", metavar="FILE",
                   help="Save audit JSON to this path (default: <output>/prediction.json)")

    args = parser.parse_args()
    if args.command == "predict":
        _cmd_predict(args)


def _cmd_predict(args: argparse.Namespace) -> None:
    if not args.ap and not args.lat:
        print("Error: provide at least --ap or --lat", file=sys.stderr)
        sys.exit(1)

    from elbow_grader import ElbowGrader, GraderConfig

    config = GraderConfig(
        confidence_threshold=args.confidence,
        run_full_lat_alignment=not args.no_alignment,
        run_sam2=not args.no_sam2,
        device=args.device,
    )

    if args.checkpoint_dir:
        grader = ElbowGrader.from_checkpoints(
            args.checkpoint_dir, config=config, sam2_ckpt=args.sam2_ckpt
        )
    else:
        grader = ElbowGrader(config=config, sam2_ckpt=args.sam2_ckpt)

    print("Model status:")
    for name, status in grader.model_status.items():
        print(f"  {name:<35} {status}")
    print()

    print("Running prediction…", flush=True)
    result = grader.predict(ap_image=args.ap, lat_image=args.lat)

    print()
    print(f"  Final grade   : {result.final_grade}")
    print(f"  CNN grade     : {result.cnn_grade}")
    print(f"  Geometric     : {result.geometric_grade}")
    conf_str = f"{result.confidence:.3f}" if result.confidence is not None else "n/a"
    print(f"  Confidence    : {conf_str}")
    print(f"  Discordant    : {result.discordant}")
    print(f"  Grade source  : {result.grade_source}")
    print(f"  OOD flagged   : {result.is_ood}")

    if result.log:
        print("\nPipeline log:")
        for line in result.log:
            print(f"  - {line}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = Path(args.json) if args.json else out_dir / "prediction.json"
    result.to_json(json_path)
    print(f"\nAudit JSON → {json_path}")

    if args.save_plots:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for CLI use
        import matplotlib.pyplot as plt

        plots = {
            "summary":        result.plot_summary,
            "gradcam":        result.plot_gradcam,
            "geometric":      result.plot_geometric,
            "cortical_width": result.plot_cortical_width,
        }
        for name, plot_fn in plots.items():
            try:
                fig = plot_fn()
                path = out_dir / f"{name}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Plot saved → {path}")
            except Exception as e:
                print(f"Warning: could not save {name} plot: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
